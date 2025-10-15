import argparse

import torch
from torchvision.datasets import CocoCaptions
import torch
import glob
import transformers

import os
import json
from tqdm import tqdm
import itertools

from .imagenet2012 import make_imagenet2012
from .templates import imagenet_templates
from .classnames import IMAGENET_CLASSNAMES
from .utils.logging import AverageMeter
from .utils.metrics import calculate_topk_accuracy
import torch.nn.functional as F
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


@torch.no_grad()
def zeroshot_classifier(model, classnames, templates, tokenizer, device, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if isinstance(classname, list):
                clsname = classname[0]
            else:
                clsname = classname
            texts = [template.format(clsname).lower() for template in templates]  # format with class

            caption_input = tokenizer(texts, max_length=args.max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)


            class_embeddings = model.get_text_features(**caption_input)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            del class_embeddings
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


@torch.no_grad()
def evaluate(args):
    assert args.naflex
    image_processor = Fgclip2ImageProcessor.from_pretrained(args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    model = FG_CLIP2_Model.from_pretrained(args.model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    device = model.device

    cur_image_size = args.image_size
    batch_size = args.batch_size
    image_folder = args.image_folder
    map_idx_file = args.map_idx_file

    if args.naflex:
        def make_image_input(image):
            return image_processor(images=image, dynamic_max_patches=False, max_num_patches=1024, return_tensors='pt')
        def _transform(n_px=224):
            return Compose( [
                            make_image_input,])
    else:
        pass
    
    dataset, dataloader, sampler = make_imagenet2012(
        transform=_transform(cur_image_size),
        batch_size=batch_size,
        root_path=None,
        image_folder=image_folder,
        training=False,
        drop_last=True,
        naflex=args.naflex>0,
        index_targets=map_idx_file
    )

    acc_top1_meter = AverageMeter()
    acc_top5_meter = AverageMeter()

    model = model.eval()
    text_features = zeroshot_classifier(model, IMAGENET_CLASSNAMES, imagenet_templates, tokenizer, device, args)


    # Evaluate
    for itr, (data) in tqdm(enumerate(dataloader)):
        # templates = SIMPLE_IMAGENET_TEMPLATES
        templates = [lambda c: f'a photo of a {c}.']
        use_format = isinstance(templates[0], str)
        def _evaluate(logits, labels, topk=(1, 5)):
            predicted_indices = torch.argmax(logits, dim=1)
            predicted_classnames = [IMAGENET_CLASSNAMES[idx] for idx in predicted_indices]
            print(f"pred labels: {predicted_classnames[:5]}")
            return calculate_topk_accuracy(logits, labels, topk=topk)
        if args.naflex:
            def _load_imgs():

                image_input = data[0].to(device, non_blocking=True)
                labels = data[-1].to(device, non_blocking=True)
                new_image_input = {}
                new_image_input["pixel_values"] = image_input["pixel_values"].squeeze(dim=1)
                new_image_input["pixel_attention_mask"] = image_input["pixel_attention_mask"].squeeze(dim=1)
                new_image_input["spatial_shapes"] = image_input["spatial_shapes"].squeeze(dim=1)
                return new_image_input, labels

            def _process_batch(image_input):
                
                with torch.no_grad():
                    image_features = model.get_image_features(**image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    logits = (model.logit_scale.exp() * image_features @ text_features)+model.logit_bias # 已转置
                    logits = torch.sigmoid(logits)
                    return logits

            image_input, labels = _load_imgs()

            logits = _process_batch(image_input)
            top1_accuracy, top5_accuracy = _evaluate(logits, labels)
            acc_top1_meter.update(top1_accuracy)
            acc_top5_meter.update(top5_accuracy)
            s = f"acc@1: {top1_accuracy * 100:.2f}%/{acc_top1_meter.avg * 100:.2f}%, acc@5: {top5_accuracy * 100:.2f}%/{acc_top5_meter.avg * 100:.2f}%"
            print(s)
            
        else:
            pass
        print(s)
       



if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument('-d', '--data-dir', default='imagenetv2-matched-frequency-format-val', type=str,
                      help='dataset path (default: None)')
    args.add_argument('-w', '--num-workers', default=8, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=128, type=int,
                      help='Batch size (default: 64)')
    args.add_argument("--model-path", type=str, default="qihoo360/fg-clip2-base")
    args.add_argument("--model-base", type=str, default="qihoo360/fg-clip2-base")
    args.add_argument("--image_folder", type=str, default="IN1K_val/val")
    args.add_argument("--map_idx_file", type=str, default="IN1K_val/imagenet2012_mapclsloc.txt")
    args.add_argument("--max_length", type=int, default=64)
    args.add_argument("--image_size", type=int, default=224)
    args.add_argument("--naflex", action='store_true', default=True)
 
    config = args.parse_args()
    evaluate(config)