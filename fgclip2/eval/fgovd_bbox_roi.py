import torch
import glob
import argparse
import os
import json
from tqdm import tqdm
import itertools
import numpy as np
from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model
from fgclip2.model.strcs.image_processing_fgclip2 import Fgclip2ImageProcessor
from transformers import AutoTokenizer

from PIL import Image
import numpy as np

def normalize_and_tensorize_boxes_naflex(bbox, image_width, image_height, real_w, real_h):
    x, y, w, h = bbox
    x1 = (x / image_width) * real_w
    y1 = (y / image_height) * real_h
    x2 = ((x + w) / image_width) * real_w
    y2 = ((y + h) / image_height) * real_h
    newbox = [[0, x1, y1, x2, y2]]
    boxes_tensor = torch.tensor(newbox, dtype=torch.float32)

    return boxes_tensor

def eval_fgovd(model, image_processor, tokenizer, device, args):
    pred_true = 0
    index_i = 0
    image_folder = args.image_folder
    ann_file = args.ann_file
    with torch.no_grad():
        with open(ann_file, 'r') as file:
            jsonlist = file.readlines()
            itemnum = len(jsonlist)

        image_size = args.image_size
        patch_size = model.config.vision_config.patch_size
        
        for item in jsonlist:
            msg = json.loads(item)
            image_path = os.path.join(image_folder, msg["img_path"])
            captions = msg["pos_expression"]
            neg_expression = msg["neg_expression"]
            captions = captions+neg_expression
            captions = [caption.lower() for caption in captions]

            boxmsg = msg["bbox"]
            bbox = (boxmsg[0],boxmsg[1],boxmsg[2],boxmsg[3])
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[0] + bbox[2])
            bottom = int(bbox[1] + bbox[3])

            img = Image.open(image_path).convert('RGB')
            image_width,image_height = img.size
            image_input = image_processor(images=img, return_tensors="pt").to(device)
            spatial_values = image_input["spatial_shapes"][0]
            real_h = spatial_values[0].item()
            real_w = spatial_values[1].item()
            boxinfo_tensor = normalize_and_tensorize_boxes_naflex(bbox,image_width,image_height,real_w,real_h)
            boxinfo_tensor = boxinfo_tensor.to(device)
            boxinfo_tensor = boxinfo_tensor.unsqueeze(dim=0)

            with torch.no_grad():
                image_features = model.get_image_box_roi_features(**image_input,box_info=boxinfo_tensor).to(device)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            caption_input = tokenizer(captions, max_length=args.max_length, padding="max_length", truncation=True, return_tensors='pt').to(device)
            text_features = model.get_text_features(**caption_input, walk_type=args.walk_type)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
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
    assert args.naflex
    image_processor = Fgclip2ImageProcessor.from_pretrained(args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    model = FG_CLIP2_Model.from_pretrained(args.model_path, device_map="cuda").cuda().eval()
    device = model.device
    
    eval_fgovd(model, image_processor, tokenizer, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--model-base", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224, help='for no-naflex siglip2')
    parser.add_argument("--naflex", action='store_true', default=True)
    parser.add_argument("--walk_type", type=str, default="box")
    parser.add_argument("--image-folder", type=str, default="data/coco/")
    parser.add_argument("--ann_file", type=str, default="")
    args = parser.parse_args()

    eval_model(args)