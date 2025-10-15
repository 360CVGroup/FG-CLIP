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
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


@torch.no_grad()
def zeroshot_classifier(model, classnames, templates, tokenizer, device, args):
    zeroshot_weights = []
    for classname in tqdm(classnames):
        if isinstance(classname, list):
            clsname = classname[0]
        else:
            clsname = classname
        texts = [clsname]

        caption_input = tokenizer(
            texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        class_embeddings = model.get_text_features(**caption_input, walk_type=args.walk_type)
        class_embedding = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
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
def eval_model(args):

    dist.init_process_group(backend='nccl')
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


    annFile = args.ann_file
    with open(annFile, 'r') as f:
        anno_info = json.load(f)

    category_list = anno_info['categories']
    image_list = anno_info['images']
    annotation_list = anno_info['annotations']

    category_names = [cat["name"] for cat in category_list]

    image_id_to_idx = {img["id"]: img["file_name"] for img in image_list}

    chunked_annotations = get_chunk(annotation_list, world_size, rank)

    text_features = zeroshot_classifier(model, category_names, imagenet_templates, tokenizer, f"cuda:{rank}", args)
    text_features = text_features.to(f"cuda:{rank}")

    image_folder = args.image_folder

    top1_correct = 0
    top5_correct = 0
    total_count = 0

    image_size = args.image_size
    patch_size = model.config.vision_config.patch_size
    feat_size = image_size // patch_size

    for anno_info in tqdm(chunked_annotations, desc=f"Rank {rank}"):
        image_id = anno_info["image_id"]
        bbox = anno_info["bbox"]
        true_category_id = anno_info['category_id'] - 1  # 调整类别索引
        image_name = image_id_to_idx[image_id]
        img_path = os.path.join(image_folder, image_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Rank {rank}: Error loading {img_path}: {e}")
            continue

        image_width, image_height = img.size
        newimg = img


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


            logits_per_text = torch.matmul(text_features, image_features.t().to(text_features.device))
            logit_scale, logit_bias = model.logit_scale.to(text_features.device), model.logit_bias.to(text_features.device)
            logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
            logits_per_image = logits_per_text.t()
            similarity = torch.sigmoid(logits_per_image).squeeze(-1).softmax(dim=-1)
        else:

            if args.naflex:
    
                image_input = image_processor(images=newimg, return_tensors="pt").to(f"cuda:{rank}")
                spatial_values = image_input["spatial_shapes"][0]
                real_h = spatial_values[0].item()
                real_w = spatial_values[1].item()
                boxinfo_tensor = normalize_and_tensorize_boxes_naflex(bbox, image_width, image_height, real_w, real_h).to(f"cuda:{rank}")
                boxinfo_tensor = boxinfo_tensor.unsqueeze(0)
            else:
                assert args.naflex
 
            image_features = model.get_image_box_roi_features(**image_input, box_info=boxinfo_tensor)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        if true_category_id in indices.cpu().numpy():
            top5_correct += 1
            if indices[0].item() == true_category_id:
                top1_correct += 1
        total_count += 1


    total_results = {
        'top1': torch.tensor(top1_correct, device=f"cuda:{rank}"),
        'top5': torch.tensor(top5_correct, device=f"cuda:{rank}"),
        'total': torch.tensor(total_count, device=f"cuda:{rank}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--walk_type", type=str, default="box")
    parser.add_argument("--naflex", action='store_true', default=True)
    parser.add_argument('--copy_head', action='store_true')
    parser.add_argument('--crop_resize', action='store_true')
    parser.add_argument("--ann_file", type=str, default='BoxClass-CN/valid_category_data_total_zh.json')
    parser.add_argument("--image-folder", type=str, default='BoxClass-CN/images')
    args = parser.parse_args()
    eval_model(args)