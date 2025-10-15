import argparse

import torch

from tqdm import tqdm
from .classes import imagenet_classes
from .data_loader import data_loader, get_label
from .templates import imagenet_templates
import torch
from torchvision.datasets import CocoCaptions
import torch
import glob
import transformers
import argparse
import os
import json
from tqdm import tqdm
import itertools

from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model
from fgclip2.model.strcs.image_processing_fgclip2_fast import Fgclip2ImageProcessorFast
from fgclip2.model.strcs.image_processing_fgclip2 import Fgclip2ImageProcessor
from transformers import AutoTokenizer

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def zeroshot_classifier(model, classnames, templates, tokenizer, args, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [(template.format(classname)).lower() for template in templates]  # format with class
 
            caption_input = tokenizer(texts, padding="max_length", max_length=args.max_length, truncation=True, return_tensors="pt").to(device)

            class_embeddings = model.get_text_features(**caption_input)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    return zeroshot_weights



def main(args):


    # Origin CLIP
    assert args.naflex
    image_processor = Fgclip2ImageProcessor.from_pretrained(args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    model = FG_CLIP2_Model.from_pretrained(args.model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    device = model.device


    if args.naflex:
        def make_image_input(image):
            return image_processor(images=image, dynamic_max_patches=False, max_num_patches=1024, return_tensors='pt')
        def _transform(n_px=224):
            return Compose( [
                            make_image_input,])
    else:
        pass


    loader, dataset = data_loader(_transform(), args) 


    zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates, tokenizer, args, device)
    total_num = 0
    true_num = 0


    with torch.no_grad():
        for i, (images, targets, paths) in enumerate(tqdm(loader)):

            images = images.to(device)

            if args.naflex:
                image_input = images.to(device, non_blocking=True)
                new_image_input = {}
                new_image_input["pixel_values"] = image_input["pixel_values"].squeeze(dim=1)
                new_image_input["pixel_attention_mask"] = image_input["pixel_attention_mask"].squeeze(dim=1)
                new_image_input["spatial_shapes"] = image_input["spatial_shapes"].squeeze(dim=1)
                image_features = model.get_image_features(**new_image_input)
            else:
                images = images.to(device)
                image_features = model.get_image_features(pixel_values=images)

            image_features /= image_features.norm(dim=-1, keepdim=True)


            logits_per_text = (
                torch.matmul(zeroshot_weights.t(), image_features.t().to(device)) * model.logit_scale.exp()
            + model.logit_bias
            )   
            logits_per_image = logits_per_text.t()

            probs = torch.sigmoid(logits_per_image) # these are the probabilities
            logits = 100.* probs

            pred = torch.argmax(logits,dim=1)
            
            total_len = pred.shape[0]
            for i in range(total_len):
                label = targets[i]
                if pred[i].item() == int(label):
                    true_num += 1
                total_num += 1

            print(true_num / total_num)
        print(true_num / total_num)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument('-d', '--data-dir', default='imagenetv2-matched-frequency-format-val', type=str,
                      help='dataset path (default: None)')
    args.add_argument('-w', '--num-workers', default=8, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=256, type=int,
                      help='Batch size (default: 64)')
    args.add_argument("--model-path", type=str, default="qihoo360/fg-clip2-base")
    args.add_argument("--model-base", type=str, default="qihoo360/fg-clip2-base")
    args.add_argument("--max_length", type=int, default=64)
    args.add_argument("--image_size", type=int, default=224)
    args.add_argument("--naflex", action='store_true', default=True)
 
    config = args.parse_args()
    main(config)
