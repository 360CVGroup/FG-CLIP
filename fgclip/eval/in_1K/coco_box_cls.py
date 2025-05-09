import torch

import glob
import transformers
import argparse
import os
import json
from tqdm import tqdm
import itertools


from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPImageProcessor,
    CLIPConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from .templates import imagenet_templates
import torch.nn.functional as F
from fgclip.model.clip_strc.fgclip import FGCLIPModel


@torch.no_grad
def zeroshot_classifier(model, classnames, templates, tokenizer, device, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if isinstance(classname, list):
                clsname = classname[0]
            else:
                clsname = classname
            texts = [template.format(clsname) for template in templates]  # format with class
 

            caption_input = torch.tensor(tokenizer(texts, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)

            walk_short_pos = True
            if args.max_length>100:
                walk_short_pos = False

            class_embeddings = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)
 
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            del class_embeddings
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def normalize_and_tensorize_boxes(bbox, image_width, image_height, feature_size=14):


    x, y, w, h = bbox
    
    # 正则化边界框坐标到0-1之间
    x1 = (x / image_width)*feature_size
    y1 = (y / image_height)*feature_size
    x2 = ((x + w) / image_width)*feature_size
    y2 = ((y + h) / image_height)*feature_size

    newbox = [[0, x1, y1, x2, y2]]
    
    # 转换为张量
    boxes_tensor = torch.tensor(newbox, dtype=torch.float32)

    return boxes_tensor

def test_clip_on_coco_boxes_base_roialign(coco_dataset, model, text_features, image_processor,device,image_size,category_id_to_idx):
    top1_correct_predictions = 0
    top5_correct_predictions = 0
    total_predictions = 0

    for image_id in tqdm(range(len(coco_dataset))):
        # 获取图像及其注释
        img, annotations = coco_dataset[image_id]
        for annotation in annotations:
            bbox = annotation['bbox']
            true_category_id = annotation['category_id']


            image_width,image_height = img.size

            if image_size==336:
                boxinfo_tensor = normalize_and_tensorize_boxes(bbox,image_width,image_height,feature_size=24)
            else:
                boxinfo_tensor = normalize_and_tensorize_boxes(bbox,image_width,image_height)


            boxinfo_tensor = boxinfo_tensor.to(device)


            image = img.resize((image_size,image_size))
  


            image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)


            with torch.no_grad():

                image_features = model.get_image_box_roi_features(image_input,box_info=boxinfo_tensor)

                # 计算文本和图像特征的相似度
                similarity = (100.0 * image_features @ text_features).softmax(dim=-1)

                # 获取最高相似度的类别索引
                values, indices = similarity[0].topk(5)

                # 比较预测类别和真实类别
                true_category_idx = category_id_to_idx.get(true_category_id)
                if true_category_idx is not None and true_category_idx in indices.cpu().numpy():
                    top5_correct_predictions += 1
                    if indices[0].item() == true_category_idx:
                        top1_correct_predictions += 1
                total_predictions += 1
                print(top1_correct_predictions," / ",total_predictions)

    top1_accuracy = top1_correct_predictions / total_predictions if total_predictions > 0 else 0
    top5_accuracy = top5_correct_predictions / total_predictions if total_predictions > 0 else 0
    return top1_accuracy, top5_accuracy



def evaluate(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    image_processor = CLIPImageProcessor.from_pretrained(args.model_base)

    interpolate_pos_encoding=False

    image_size = args.image_size
    batch_size = args.batch_size


    coco_dataset = CocoDetection(
        root=args.image_folder+'/val2017',
        annFile=args.image_folder+'/annotations/instances_val2017.json',
    )

    model = FGCLIPModel.from_pretrained(args.model_path,ignore_mismatched_sizes=True).cuda()
    model = model.eval()

    device = model.device


    category_ids = list(coco_dataset.coco.cats.keys())
    category_names = [coco_dataset.coco.cats[cat_id]['name'] for cat_id in category_ids]

    category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    text_features = zeroshot_classifier(model, category_names, imagenet_templates, tokenizer, device, args)

    top1_accuracy, top5_accuracy = test_clip_on_coco_boxes_basecrop(coco_dataset, model, text_features,image_processor,device,image_size,category_id_to_idx)
    print(f"Top-1 Classification accuracy: {top1_accuracy:.2%}")
    print(f"Top-5 Classification accuracy: {top5_accuracy:.2%}")



if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument('-w', '--num-workers', default=8, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=256, type=int,
                      help='Batch size (default: 64)')
    args.add_argument("--model-path", type=str, default="qihoo360/fg-clip-base")
    args.add_argument("--model-base", type=str, default="qihoo360/fg-clip-base")
    args.add_argument("--max_length", type=int, default=77)
    args.add_argument("--image-folder", type=str, default="data/coco")
    args.add_argument("--image_size", type=int, default=224)
 
    config = args.parse_args()
    evaluate(config)