import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu


from torchvision.datasets import CocoCaptions

import glob
import transformers
import argparse
import os
import json
from tqdm import tqdm
import itertools
import numpy as np

from fgclip.model.clip_strc.fgclip import FGCLIPModel
from torch import nn, einsum

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPImageProcessor,
    CLIPConfig,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def normalize_and_tensorize_boxes(bbox, image_width, image_height, feature_size=14):


    x, y, w, h = bbox
    
    # 正则化边界框坐标到0-1之间
    x1 = (x / image_width)*feature_size
    y1 = (y / image_height)*feature_size
    x2 = ((w) / image_width)*feature_size
    y2 = ((h) / image_height)*feature_size

    newbox = [[0, x1, y1, x2, y2]]
    
    # 转换为张量
    boxes_tensor = torch.tensor(newbox, dtype=torch.float32)

    return boxes_tensor


def eval_fg(model,image_processor,tokenizer,device,args):
    
    pred_true = 0
    index_i = 0

    with torch.no_grad():

        with open('fgovd_json/1_attributes_llava.jsonl', 'r') as file:

            jsonlist = file.readlines()
            itemnum = len(jsonlist) 

        image_size = args.image_size
        

        for item in jsonlist:

            msg = json.loads(item)

            image_path = args.image_folder+msg["img_path"]
            # captions = []
            captions = msg["pos_expression"]

            neg_expression = msg["neg_expression"]
            captions = captions+neg_expression

            boxmsg = msg["bbox"]
            bbox = (boxmsg[0],boxmsg[1],boxmsg[0]+boxmsg[2],boxmsg[1]+boxmsg[3])


            img = Image.open(image_path).convert('RGB')

            image_width,image_height = img.size

            if image_size==336:
                boxinfo_tensor = normalize_and_tensorize_boxes(bbox,image_width,image_height,feature_size=24)
            else:
                boxinfo_tensor = normalize_and_tensorize_boxes(bbox,image_width,image_height)

            boxinfo_tensor = boxinfo_tensor.to(device)

            image = img.resize((image_size,image_size))
 
            image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

            image_features = model.get_image_box_roi_features(image_input,box_info=boxinfo_tensor)

            caption_input = torch.tensor(tokenizer(captions, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
   
            walk_short_pos = True

            if args.max_length>100:
                walk_short_pos = False

            text_features = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)


            max_value = torch.max(similarity[0])


            value_at_index_0 = similarity[0][0]

            is_max_at_index_0 = torch.equal(max_value, value_at_index_0)

            if is_max_at_index_0:
                pred_true+=1
            else:
                pass

            index_i+=1
            print(index_i," / ", itemnum, "   precision: ", pred_true/itemnum)



def eval_model(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_base)

    model = FGCLIPModel.from_pretrained(args.model_path).cuda()

    model.eval()
    device = model.device

    eval_fg(model,image_processor,tokenizer,device,args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--model-base", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--image-folder", type=str, default="path of coco")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    eval_model(args)
