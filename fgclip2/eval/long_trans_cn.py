import torch
import glob
import argparse
import os
import json

from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model
from fgclip2.model.strcs.image_processing_fgclip2_fast import Fgclip2ImageProcessorFast
from fgclip2.model.strcs.image_processing_fgclip2 import Fgclip2ImageProcessor
from transformers import AutoTokenizer
from PIL import Image


def eval_long_cn(model,image_processor,tokenizer,device,args):

    anno_path = args.ann_file
    fp = open(anno_path, "r")
    lines = fp.readlines()
    image_features = []
    text_features = []
    pred_true = 0
    image_size = args.image_size
    with torch.no_grad():
        index = 0

        for line in lines:
            image_name, caption = line.rstrip().split("\t")
            image = Image.open(image_name).convert("RGB")

            
            image_input = image_processor(images=image, return_tensors='pt').to(device)

            image_feature = model.get_image_features(**image_input)
            image_features.append(image_feature)

            caption_input = tokenizer(caption, max_length=args.max_length, padding="max_length", truncation=True, return_tensors='pt').to(device)


            text_feature = model.get_text_features(**caption_input,walk_type=args.walk_type)
            text_features.append(text_feature)
            index+=1

            print(index,": ", len(lines))

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features.squeeze() @ text_features.squeeze().T
        similarity = torch.sigmoid((model.logit_scale.exp()*similarity)+model.logit_bias)


        captionnums = len(lines)
        
    
        print("I2T")
        for i in range(captionnums):
            pred = similarity[i]
            b = pred.argsort()[-1:]

            true_index = i
            if b == true_index:
                pred_true = pred_true + 1

        print(pred_true / captionnums)

        pred_true = 0

        print("T2I")
        similarity = similarity.T
        for i in range(captionnums):
            pred = similarity[i]
            b = pred.argsort()[-1:]

            true_index = i
            if b == true_index:
                pred_true = pred_true + 1

        print(pred_true/captionnums)


def eval_model(args):
    assert args.naflex
    image_processor = Fgclip2ImageProcessor.from_pretrained(args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    model = FG_CLIP2_Model.from_pretrained(args.model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    device = model.device

    eval_long_cn(model,image_processor,tokenizer,device,args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--model-base", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--max_length", type=int, default=196)
    parser.add_argument("--image-folder", type=str, default="coco")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--walk_type", type=str, default="long")
    parser.add_argument("--naflex", action='store_true', default=True)
    parser.add_argument("--ann_file", type=str, default="")
    args = parser.parse_args()

    eval_model(args)