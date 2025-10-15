import torch
import torch.distributed as dist
import argparse
import os
import json
from tqdm import tqdm
import math

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection

from .templates import imagenet_templates
import torch.nn.functional as F

from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model
from fgclip2.model.strcs.image_processing_fgclip2_fast import Fgclip2ImageProcessorFast
from fgclip2.model.strcs.image_processing_fgclip2 import Fgclip2ImageProcessor
from transformers import AutoTokenizer


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


@torch.no_grad()
def zeroshot_classifier(model, classnames, templates, tokenizer, device, args):
    zeroshot_weights = []
    for classname in tqdm(classnames, desc="Building text features", disable=(dist.get_rank() != 0)):
        if isinstance(classname, list):
            clsname = classname[0]
        else:
            clsname = classname
        texts = [template.format(clsname).lower() for template in templates]

        caption_input = tokenizer(
            texts,
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        class_embeddings = model.get_text_features(**caption_input, walk_type=args.walk_type)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
        del class_embeddings

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def normalize_and_tensorize_boxes(bbox, image_width, image_height, feature_size=14, roi_align=True):
    x, y, w, h = bbox
    if roi_align:
        x1 = (x / image_width) * feature_size
        y1 = (y / image_height) * feature_size
        x2 = ((x + w) / image_width) * feature_size
        y2 = ((y + h) / image_height) * feature_size
        newbox = [[0, x1, y1, x2, y2]]
    else:
        x1 = x / image_width
        y1 = y / image_height
        x2 = (x + w) / image_width
        y2 = (y + h) / image_height
        newbox = [[x1, y1, x2, y2]]
    return torch.tensor(newbox, dtype=torch.float32)


def normalize_and_tensorize_boxes_naflex(bbox, image_width, image_height, real_w, real_h):
    x, y, w, h = bbox
    x1 = (x / image_width) * real_w
    y1 = (y / image_height) * real_h
    x2 = ((x + w) / image_width) * real_w
    y2 = ((y + h) / image_height) * real_h
    newbox = [[0, x1, y1, x2, y2]]
    return torch.tensor(newbox, dtype=torch.float32)




@torch.no_grad()
def evaluate(args):

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    assert args.naflex

    image_processor = Fgclip2ImageProcessor.from_pretrained(args.model_base)

    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    model = FG_CLIP2_Model.from_pretrained(args.model_path).to(device).eval()

    if args.copy_head:
        model.copy_dense_feature_head()


    coco_dataset = CocoDetection(
        root=args.image_folder,
        annFile=args.ann_file,
    )


    category_ids = list(coco_dataset.coco.cats.keys())
    category_names = [coco_dataset.coco.cats[cat_id]['name'] for cat_id in category_ids]
    category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}


    text_features = zeroshot_classifier(model, category_names, imagenet_templates, tokenizer, device, args)


    indices = list(range(len(coco_dataset)))
    chunked_indices = get_chunk(indices, world_size, rank)


    top1_correct = 0
    top5_correct = 0
    total_predictions = 0


    image_size = args.image_size
    patch_size = model.config.vision_config.patch_size
    feat_size = image_size // patch_size


    for idx in tqdm(chunked_indices, desc=f"Rank {rank} processing"):
        try:
            img, annotations = coco_dataset[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            image_width, image_height = img.size

            for annotation in annotations:
                try:
                    bbox = annotation['bbox']
                    true_category_id = annotation['category_id']
                    true_category_idx = category_id_to_idx.get(true_category_id)
                    if true_category_idx is None:
                        continue  # 忽略无效类别
                    
                    if args.crop_resize:
                        x, y, width, height = map(int, bbox)
                        cropped_img = img.crop((x, y, x + width, y + height))
    
                        try:
                            newimg = cropped_img
                            image_input = image_processor(images=newimg, return_tensors="pt").to(device)
                        except:
                            newimg = cropped_img.resize((image_size,image_size))
                            image_input = image_processor(images=newimg, return_tensors="pt").to(device)

                        image_features = model.get_image_features(**image_input)
                        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
          
                        logits_per_text = torch.matmul(text_features.t(), image_features.t().to(text_features.device))
                        logit_scale, logit_bias = model.logit_scale.to(text_features.device), model.logit_bias.to(text_features.device)
                        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
                        logits_per_image = logits_per_text.t()
                        similarity = torch.sigmoid(logits_per_image).squeeze(-1).softmax(dim=-1)

                    else:
                        newimg = img  
                        
                        if args.naflex:

                            image_input = image_processor(
                                images=newimg,
                                return_tensors="pt"
                            ).to(device)

                            spatial_values = image_input["spatial_shapes"][0]
                            real_h = spatial_values[0].item()
                            real_w = spatial_values[1].item()

                            boxinfo_tensor = normalize_and_tensorize_boxes_naflex(
                                bbox, image_width, image_height, real_w, real_h
                            ).to(device).unsqueeze(0)
                        else:
                            assert args.naflex

                        image_features = model.get_image_box_roi_features(**image_input, box_info=boxinfo_tensor)
                        similarity = (100.0 * image_features @ text_features).softmax(dim=-1)


                    _, indices_pred = similarity[0].topk(5)

                    if true_category_idx in indices_pred.cpu().numpy():
                        top5_correct += 1
                        if indices_pred[0].item() == true_category_idx:
                            top1_correct += 1
                    total_predictions += 1

                except Exception as e:
                    print(f"Rank {rank}: Error processing annotation in image {idx}: {e}")
                    continue 

        except Exception as e:
            print(f"Rank {rank}: Error loading image {idx}: {e}")
            continue  

        if rank == 0 and total_predictions % 100 == 0:
            print(f"Processed {total_predictions} boxes")


    total_results = {
        'top1': torch.tensor(top1_correct, device=device),
        'top5': torch.tensor(top5_correct, device=device),
        'total': torch.tensor(total_predictions, device=device)
    }

    dist.barrier()
    dist.all_reduce(total_results['top1'], op=dist.ReduceOp.SUM)
    dist.all_reduce(total_results['top5'], op=dist.ReduceOp.SUM)
    dist.all_reduce(total_results['total'], op=dist.ReduceOp.SUM)

    if rank == 0:
        top1_acc = total_results['top1'].item() / total_results['total'].item()
        top5_acc = total_results['top5'].item() / total_results['total'].item()
        print(f"Top-1 Accuracy: {top1_acc:.4f} ({total_results['top1'].item()}/{total_results['total'].item()})")
        print(f"Top-5 Accuracy: {top5_acc:.4f} ({total_results['top5'].item()}/{total_results['total'].item()})")

    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP inference on COCO with DDP')
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--walk_type", type=str, default="box")
    parser.add_argument("--naflex", action='store_true', default=True)
    parser.add_argument('--copy_head', action='store_true')
    parser.add_argument('--crop_resize', action='store_true')
    parser.add_argument("--ann_file", type=str, default="coco/annotations/instances_val2017.json")
    parser.add_argument("--image-folder", type=str, default="coco/val2017")
    args = parser.parse_args()

    evaluate(args)