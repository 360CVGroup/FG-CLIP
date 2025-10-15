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

from transformers import AutoProcessor, AutoModel,Siglip2ImageProcessor
from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model


def normalize_and_tensorize_boxes(bbox, image_width, image_height, feature_size=14):
    x, y, w, h = bbox
    x1 = (x / image_width)*feature_size
    y1 = (y / image_height)*feature_size
    x2 = ((w) / image_width)*feature_size
    y2 = ((h) / image_height)*feature_size
    newbox = [[0, x1, y1, x2, y2]]
    boxes_tensor = torch.tensor(newbox, dtype=torch.float32)

    return boxes_tensor


def normalize_and_tensorize_boxes_naflex(bbox, image_width, image_height, real_w, real_h):
    x, y, w, h = bbox
    x1 = (x / image_width)*real_w
    y1 = (y / image_height)*real_h
    x2 = ((w) / image_width)*real_w
    y2 = ((h) / image_height)*real_h

    newbox = [[0, x1, y1, x2, y2]]
    boxes_tensor = torch.tensor(newbox, dtype=torch.float32)

    return boxes_tensor

def determine_max_value(image):
    w, h = image.size
    max_val = (w // 16) * (h // 16)
    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128


def eval_coco(model,image_processor,tokenizer,device,args):
    
    pred_true = 0
    index_i = 0
    image_size = args.image_size
    patch_size = model.config.vision_config.patch_size
    feat_size = image_size // patch_size

    with torch.no_grad():

        with open('/hbox2dir/fgovd_json/shuffle_negatives_llava.jsonl', 'r') as file:
        # with open('/hbox2dir/fgovd_json/1_attributes_llava.jsonl', 'r') as file:
        # with open('/hbox2dir/fgovd_json/2_attributes_llava.jsonl', 'r') as file:
        # with open('/hbox2dir/fgovd_json/3_attributes_llava.jsonl', 'r') as file:
            jsonlist = file.readlines()
            itemnum = len(jsonlist)  # 计算行数


        image_size = args.image_size
        

        for item in jsonlist:

            msg = json.loads(item)

            image_path = "coco/"+msg["img_path"]
            captions = msg["pos_expression"]

            neg_expression = msg["neg_expression"]
            captions = captions+neg_expression
            captions = [cap.lower() for cap in captions]


            caption_input = tokenizer(captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
 
            walk_short_pos = True
            if args.max_length>100:
                walk_short_pos = False

            if args.naflex:
                text_features = model.get_text_features(**caption_input,use_bbox=args.use_box, walk_short_pos=walk_short_pos)
            else:
                text_features = model.get_text_features(**caption_input,walk_short_pos=walk_short_pos)

            text_features = text_features/ text_features.norm(p=2, dim=-1, keepdim=True)
            
            boxmsg = msg["bbox"]
            bbox = (boxmsg[0],boxmsg[1],boxmsg[2],boxmsg[3])
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[0] + bbox[2])
            bottom = int(bbox[1] + bbox[3])


            img = Image.open(image_path).convert('RGB')

            image_width,image_height = img.size

            if args.crop_resize:
                cropped_img = img.crop((left, top, right, bottom))

                try:
                    newimg = cropped_img
                    if args.naflex:
                        max_patches = determine_max_value(newimg)
                        image_input = image_processor(images=newimg, max_num_patches=max_patches, return_tensors="pt").to(device)
                    else:
                        image_input = image_processor(images=newimg, return_tensors="pt").to(device)
                except:
                    newimg = cropped_img.resize((image_size,image_size))
                    if args.naflex:
                        max_patches = determine_max_value(newimg)
                        image_input = image_processor(images=newimg, max_num_patches=max_patches, return_tensors="pt").to(device)
                    else:
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
                    image_input = image_processor(images=img, max_num_patches=determine_max_value(img),return_tensors="pt").to(device)

                    spatial_values = image_input["spatial_shapes"][0]
                    real_h = spatial_values[0].item()
                    real_w = spatial_values[1].item()
                    boxinfo_tensor = normalize_and_tensorize_boxes_naflex(bbox,image_width,image_height,real_w,real_h)
                    boxinfo_tensor = boxinfo_tensor.to(device)
                    boxinfo_tensor = boxinfo_tensor.unsqueeze(dim=0)
                else:
                    boxinfo_tensor = normalize_and_tensorize_boxes(bbox,image_width,image_height,feat_size)
                    boxinfo_tensor = boxinfo_tensor.to(device)
                    image_input = image_processor(images=img, return_tensors="pt").to(device)

                image_features = model.get_image_box_roi_features(**image_input,box_info=boxinfo_tensor)

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
    image_processor = Siglip2ImageProcessor.from_pretrained(args.model_base)
    model = FG_CLIP2_Model.from_pretrained(args.model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    device = model.device

    if args.copy_head:
        # NOTE only for origin SigLIP2 test
        model.copy_dense_feature_head()

    device = model.device
    eval_coco(model,image_processor,tokenizer,device,args)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--model-base", type=str, default="qihoo360/fg-clip2-base")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--image-folder", type=str, default="coco")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--naflex", action='store_true', default=True)
    parser.add_argument('--use_box', action='store_true')
    parser.add_argument('--copy_head', action='store_true')
    parser.add_argument('--crop_resize', action='store_true')
    args = parser.parse_args()

    eval_model(args)