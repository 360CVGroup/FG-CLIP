import argparse
# import sys
# sys.path.append('../../..')
# from model import longclip
import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
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
            texts = [template.format(classname) for template in templates]  # format with class

            caption_input = torch.tensor(tokenizer(texts, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)

            walk_short_pos = True

            if args.max_length>100:
                walk_short_pos = False

            class_embeddings = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def main(args):

    softmax = torch.nn.Softmax(dim=1)
    # Origin CLIP
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_base)



    cur_image_size = args.image_size


    def make_image_input(image):
        return image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    def _transform(n_px=224):
        return Compose( [
                        Resize((n_px,n_px), interpolation=BICUBIC),
                        make_image_input,])

        loader, dataset = data_loader(_transform(cur_image_size), args)   

    model = FGCLIPModel.from_pretrained(args.model_path,ignore_mismatched_sizes=True).cuda()


    device = model.device
    
    zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates, tokenizer, args, device)
    total_num = 0
    true_num = 0


    with torch.no_grad():
        for i, (images, targets, paths) in enumerate(tqdm(loader)):

            images = images.to(device)

            image_features = model.get_image_features(images,interpolate_pos_encoding=interpolate_pos_encoding)

            image_features /= image_features.norm(dim=-1, keepdim=True)



            logits = 100. * image_features @ zeroshot_weights
            logits = softmax(logits)
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
    args.add_argument('-d', '--data-dir', default='data/imagenetv2-matched-frequency-format-val', type=str,
                      help='dataset path (default: None)')
    args.add_argument('-w', '--num-workers', default=8, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=256, type=int,
                      help='Batch size (default: 64)')
    args.add_argument("--model-path", type=str, default="qihoo360/fg-clip-base")
    args.add_argument("--model-base", type=str, default="qihoo360/fg-clip-base")
    args.add_argument("--max_length", type=int, default=77)
    args.add_argument("--image_size", type=int, default=224)
 
    config = args.parse_args()
    main(config)
