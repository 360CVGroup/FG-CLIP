import torch
import re
import json
import glob
import argparse
import os
import json

from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model
from fgclip2.model.strcs.image_processing_fgclip2_fast import Fgclip2ImageProcessorFast
from fgclip2.model.strcs.image_processing_fgclip2 import Fgclip2ImageProcessor
from transformers import AutoTokenizer

from PIL import Image


def get_pairs(args):

    pattern = re.compile(r'COCO_(train|val)2014_(\d{12})#zhm#\d\s+(.*)')
    image_to_texts_dict = {}
    with open(args.ann_file, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:

                image_id = f"COCO_{match.group(1)}2014_{match.group(2)}.jpg"

                text_description = match.group(3)

                if image_id not in image_to_texts_dict:
                    image_to_texts_dict[image_id] = []
                image_to_texts_dict[image_id].append(text_description)

    return image_to_texts_dict


def eval_coco_cn(model,processor,tokenizer,device,args):
    root_path = args.image_folder
    image_features = []
    text_features = []
    pred_true = 0


    pair = get_pairs(args)
    with torch.no_grad():
        index = 0
        # nextcaptions = coco[1][1]
        for imagename, captions in pair.items():

            if "train" in imagename:
                fullname = root_path+"/train2014/"+imagename
            elif "val" in imagename:
                fullname = root_path+"/val2014/"+imagename
            else:
                fullname = root_path+"/test2014/"+imagename

            image = Image.open(fullname).convert("RGB")


            if args.naflex:
                inputs = processor(images=image, return_tensors="pt").to(device)
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)


            image_feature = model.get_image_features(**inputs)
            image_features.append(image_feature)
            captions = [caption.lower() for caption in captions[0:5]]

            caption_input = tokenizer(captions, padding="max_length", max_length=args.max_length, truncation=True, return_tensors="pt").to(device)

            text_feature = model.get_text_features(**caption_input)
            
            text_features.extend(text_feature)
            index+=1

            print(index,": ", len(pair))

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        

        logits_per_text = torch.matmul(text_features, image_features.t().to(text_features.device))
        logit_scale, logit_bias = model.logit_scale.to(text_features.device), model.logit_bias.to(text_features.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        logits_per_image = logits_per_text.t()
        similarity = torch.sigmoid(logits_per_image).squeeze(-1)

        

        print("I2T")
        for i in range(1000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 1000)
        pred_true = 0

        for i in range(1000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 1000)
        pred_true = 0

        for i in range(1000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 1000)
        pred_true = 0

        print("T2I")
        similarity = similarity.T
        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/5000)
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/5000)
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/5000)






def evaluate(args):
    assert args.naflex
    image_processor = Fgclip2ImageProcessor.from_pretrained(args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    model = FG_CLIP2_Model.from_pretrained(args.model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    device = model.device
    eval_coco_cn(model,image_processor,tokenizer,device,args)



if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument("--model-path", type=str, default="qihoo360/fg-clip2-base")
    args.add_argument("--model-base", type=str, default="qihoo360/fg-clip2-base")
    args.add_argument("--max_length", type=int, default=64)
    args.add_argument("--image_size", type=int, default=224, help='for no-naflex siglip2')
    args.add_argument("--image-folder", type=str, default="data/coco")
    args.add_argument("--naflex", action='store_true', default=True)
    args.add_argument("--ann_file", type=str, default="")
 
    config = args.parse_args()
    evaluate(config)

