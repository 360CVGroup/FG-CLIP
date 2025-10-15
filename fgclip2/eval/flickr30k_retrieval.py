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
import numpy as np



def eval_flickr(model, ann_file, image_processor,tokenizer,device,args):
    image_features = []
    text_features = []
    pred_true = 0
    image_size = args.image_size

    dataset = json.load(open(ann_file))
    print(len(dataset), type(dataset))
    with torch.no_grad():
        index = 0
        for i in range(len(dataset)):
            image_name = dataset[i]["image"]
            captions = dataset[i]["caption"]
            image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")


            inputs = image_processor(images=image, return_tensors="pt").to(device)


            image_feature = model.get_image_features(**inputs)
            image_features.append(image_feature)

            captions = [cap.lower() for cap in captions]
            caption_input = tokenizer(captions, padding="max_length", max_length=args.max_length, truncation=True, return_tensors="pt").to(device)

            text_feature = model.get_text_features(**caption_input)
            
            text_features.extend(text_feature)
            index+=1

            print(index,": ", len(dataset))

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits_per_text = torch.matmul(text_features, image_features.t().to(text_features.device))
        logit_scale, logit_bias = model.logit_scale.to(text_features.device), model.logit_bias.to(text_features.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        logits_per_image = logits_per_text.t()
        similarity = torch.sigmoid(logits_per_image).squeeze(-1)


        image_count = image_features.shape[0]
        text_count = text_features.shape[0]
        print("I2T")
        for i in range(image_count):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / image_count)
        pred_true = 0

        for i in range(image_count):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / image_count)
        pred_true = 0

        for i in range(image_count):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / image_count)
        pred_true = 0

        print("T2I")
        similarity = similarity.T
        for i in range(text_count):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/text_count)
        pred_true = 0

        for i in range(text_count):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/text_count)
        pred_true = 0

        for i in range(text_count):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/text_count)
  

def eval_model(args):

    assert args.naflex
    image_processor = Fgclip2ImageProcessor.from_pretrained(args.model_base)
    model = FG_CLIP2_Model.from_pretrained(args.model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    device = model.device
    model.eval()

    ann_file = args.ann_file
    eval_flickr(model,ann_file,image_processor,tokenizer,device,args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--model-base", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--image-folder", type=str, default="flickr30k/")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--naflex", action='store_true', default=True)
    parser.add_argument("--ann_file", type=str, default="flickr30k/flickr30k_test.json")
    args = parser.parse_args()

    eval_model(args)