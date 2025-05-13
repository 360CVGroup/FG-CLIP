import torch


import torch_npu
from torch_npu.contrib import transfer_to_npu


from torchvision.datasets import CocoCaptions

import glob
import transformers
import argparse
import os
import json
from tqdm import tqdm
import itertools
from torch import nn, einsum
from fgclip.model.clip_strc.fgclip import FGCLIPModel

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPImageProcessor,
    SiglipImageProcessor,
    CLIPConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


from PIL import Image, ImageDraw



def eval_1k(model,image_processor,tokenizer,device,args):
    image_features = []
    text_features = []
    pred_true = 0
    image_size = args.image_size
    with torch.no_grad():
        index = 0

        

        with open(args.jsonfile_path, 'r',encoding='utf8')as fp:
            json_data = json.load(fp)[:1000]


        for json_item in json_data:
            caption = json_item['conversations'][1]['value']
            lastname = json_item['image']

            image_name = args.image_folder+lastname.split("/")[-1]
            image = Image.open(image_name).convert("RGB")


            image = image.resize((image_size,image_size))

 
            image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)


            image_feature = model.get_image_features(image_input)


            image_features.append(image_feature)

            captions = [caption]
            
            caption_input = torch.tensor(tokenizer(captions, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)

            walk_short_pos = True

            if args.max_length>100:
                walk_short_pos = False


            text_feature = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)
            
            text_features.extend(text_feature)
            index+=1


            print(index,": ", len(json_data))

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features.squeeze() @ text_features.squeeze().T


        captionnums = len(json_data)
        
    
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_base)


    image_processor = CLIPImageProcessor.from_pretrained(args.model_base)
    model = FGCLIPModel.from_pretrained(args.model_path).cuda()

    model.eval()

    device = model.device
    eval_1k(model,image_processor,tokenizer,device,args)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qihoo360/fg-clip-base")
    parser.add_argument("--model-base", type=str, default="qihoo360/fg-clip-base")
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--image-folder", type=str, default="/mm-datasets/public/coco")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--jsonfile_path", type=str, default="/hbox2dir/share-captioner_coco_lcs_sam_1246k_1107.json")

    
    args = parser.parse_args()

    eval_model(args)