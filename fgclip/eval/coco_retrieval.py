import torch

from torchvision.datasets import CocoCaptions

import glob
import transformers
import argparse
import os
import json
from tqdm import tqdm
import itertools
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from fgclip.model.clip_strc.fgclip import FGCLIPModel

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

# from PIL import Image
from PIL import Image, ImageDraw
import numpy as np

def eval_coco(model, coco,image_processor,tokenizer,device,args):
    image_features = []
    text_features = []
    pred_true = 0
    image_size = args.image_size
    with torch.no_grad():
        index = 0
        # nextcaptions = coco[1][1]
        for image, captions in coco:
            image = image.resize((image_size,image_size))

            image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

            image_feature = model.get_image_features(image_input)
            image_features.append(image_feature)

   
            captions = captions[0:5]

            caption_input = torch.tensor(tokenizer(captions, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
  
            walk_short_pos = True

            if args.max_length>100:
                walk_short_pos = False

            text_feature = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)
            
            text_features.extend(text_feature)
            index+=1

            print(index,": ", len(coco))

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features.squeeze() @ text_features.squeeze().T

    
        print("I2T")
        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 5000)
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 5000)
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 5000)
        pred_true = 0

        print("T2I")
        similarity = similarity.T
        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/25000)
        pred_true = 0

        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/25000)
        pred_true = 0

        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/25000)


      


def eval_model(args):
    # Model

    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_base)

    model = FGCLIPModel.from_pretrained(args.model_path).cuda()

    model.eval()
        
    coco = CocoCaptions(root=args.image_folder+"/val2017/", annFile=args.image_folder+"/annotations/captions_val2017.json", transform=None)
    device = model.device
    eval_coco(model,coco,image_processor,tokenizer,device,args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qihoo360/fg-clip-base")
    parser.add_argument("--model-base", type=str, default="qihoo360/fg-clip-base")
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--image-folder", type=str, default="path of coco")
    parser.add_argument("--image_size", type=int, default=224)
    
    args = parser.parse_args()

    eval_model(args)