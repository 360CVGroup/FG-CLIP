import argparse
# import sys
# sys.path.append('../../..')
# from model import longclip
import torch
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
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
# from myclip.model.clip_strc.new_clip_r import LongCLIPModel
# from myclip.model.clip_strc.myclip_box import LongCLIPModel
# from myclip.model.clip_strc.myclip_filip import LongCLIPModel
from myclip.model.clip_strc.myclip_filip import LongCLIPModel

# from myclip.model.siglip_strc.mysiglip import LongSiglipModel
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    SiglipModel,
    SiglipImageProcessor,
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
        # zeroshot_weights = []
        # for classname in tqdm(classnames):
        #     texts = [template.format(classname) for template in templates]  # format with class
        #     # texts = [classname for template in templates]  # format with class
            
        #     caption_input = torch.tensor(tokenizer(texts, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
        #     # caption_input = tokenizer(captions, padding=True, return_tensors="pt").to(device)
        #     class_embeddings,s_text_token_feas = model.get_text_features(caption_input,walk_short_pos=True)
        #     # texts = longclip.tokenize(texts).cuda()  # tokenize
        #     # class_embeddings = model.encode_text(texts)  # embed with text encoder
        #     # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        #     # class_embedding = class_embeddings.mean(dim=0)
        #     # class_embedding /= class_embedding.norm()
        #     # zeroshot_weights.append(class_embedding)
        #     s_text_token_feas /= s_text_token_feas.norm(dim=-1, keepdim=True)
        #     class_embedding = s_text_token_feas.mean(dim=0)
        #     class_embedding /= class_embedding.norm()
        #     zeroshot_weights.append(class_embedding)
            # print(class_embedding.shape)
            
        # zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        # text_mask = textinput != tokenizer.pad_token_id

        caption_input = torch.tensor(tokenizer(classnames, max_length=args.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
        # caption_input = tokenizer(captions, padding=True, return_tensors="pt").to(device)
        class_embeddings,s_text_token_feas = model.get_text_features(caption_input,walk_short_pos=True)
        # texts = longclip.tokenize(texts).cuda()  # tokenize
        # class_embeddings = model.encode_text(texts)  # embed with text encoder
        s_text_token_feas /= s_text_token_feas.norm(dim=-1, keepdim=True)
        zeroshot_weights = s_text_token_feas
        text_mask = caption_input != tokenizer.pad_token_id

    return zeroshot_weights,text_mask

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def main(args):

    softmax = torch.nn.Softmax(dim=1)
    # Origin CLIP
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_base)
    # image_processor = SiglipImageProcessor.from_pretrained(args.model_base)

    def make_image_input(image):
        return image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    def _transform(n_px=224):
        return Compose([Resize((n_px,n_px), interpolation=BICUBIC),
                        make_image_input,])

    loader, dataset = data_loader(_transform(224), args)   
        # return Compose([
        #     Resize((n_px,n_px), interpolation=BICUBIC),
        #     # CenterCrop(n_px),
        #     _convert_image_to_rgb,
        #     ToTensor(),
        #     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])
    # config = CLIPConfig.from_pretrained(args.model_path)
    # config.text_config.max_position_embeddings = args.max_length
    # Origin CLIP
    # model = CLIPModel.from_pretrained(args.model_path).cuda()
    model = LongCLIPModel.from_pretrained(args.model_path,ignore_mismatched_sizes=True).cuda()
    model.copy_weight()
    # model = SiglipModel.from_pretrained(args.model_base).cuda()
    # model = LongSiglipModel.from_pretrained(args.model_path).cuda()
    # model.eval()

    device = model.device
    
    zeroshot_weights,text_mask = zeroshot_classifier(model, imagenet_classes, imagenet_templates, tokenizer, args, device)
    total_num = 0
    true_num = 0

    # print(zeroshot_weights.shape)
    # print(text_mask.shape)
    # [1000, 77, 512]
    # breakpoint()
    # 768*1000

    with torch.no_grad():
        for i, (images, targets, paths) in enumerate(tqdm(loader)):
            # print(targets)
            # breakpoint()
            images = images.to(device)
            # print(images.shape)

            # image_input = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(device)
            image_features,vision_nocls_token_feas = model.get_image_features(images)

            # predict
            # image_features = model.encode_image(images)
            vision_nocls_token_feas /= vision_nocls_token_feas.norm(dim=-1, keepdim=True)
            # print(vision_nocls_token_feas.shape)
            # [256, 49, 512]
            # breakpoint()
            image_features = vision_nocls_token_feas
            text_features = zeroshot_weights
            # text_mask = text_mask.cpu()

            sim_text_to_image = einsum('x t d, y i d -> x y t i', text_features, image_features) 
            # sim_image_to_text = einsum('x t d, y i d -> x y t i', image_features, text_features) 
            # print(sim_text_to_image[0,0])
            # print(sim_image_to_text[0,0])
            # import pdb
            # pdb.set_trace()
            
            

            sim_image_to_text = sim_text_to_image
            # if self.extra_latent_projection:
            #     sim_image_to_text = einsum('m x t d, n y i d -> m n x y t i', text_latents_extra, image_latents_extra)

            text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
            # print(text_to_image.shape)
            text_to_image_mask = rearrange(text_mask, 'b t -> b 1 t')
            text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)
            # print(text_to_image.shape)

            image_to_text_mask = rearrange(text_mask, 'b t -> b 1 t 1')
            masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
            image_to_text = reduce(reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')

            image_to_text = image_to_text.T

            print(image_to_text.shape)

            # print(image_to_text.shape)
            # breakpoint()
            # print(image_features.shape)
            # 256*768

            # logits_per_text = (
            #     torch.matmul(zeroshot_weights.t(), image_features.t().to(device)) * model.logit_scale.exp()
            # + model.logit_bias
            # )   
            # logits_per_image = logits_per_text.t()

            # # logits_per_image = outputs.logits_per_image
            # probs = torch.sigmoid(logits_per_image) # these are the probabilities
            # print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
            # print(probs)
            # 256*1000
            # breakpoint()
            # pred = probs
            # logits = 100.* probs


            # logits = 100. * image_features @ zeroshot_weights
            logits = 100. * image_to_text
            logits = softmax(logits)
            pred = torch.argmax(logits,dim=1)
            
            total_len = pred.shape[0]
            for i in range(total_len):
                label = targets[i]
                # print(pred[i].item(),label.item())
                # print(type(label),type(pred[i].item()),type(label.item()))
                # label = get_label(label).item()
                if pred[i].item() == int(label):
                    true_num += 1
                total_num += 1

            # breakpoint()
            
            #save_to_file(logits, targets, paths)
        print(true_num / total_num)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument('-d', '--data-dir', default='/mm-datasets/public/imagenetv2-matched-frequency-format-val', type=str,
                      help='dataset path (default: None)')
    args.add_argument('-w', '--num-workers', default=8, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=256, type=int,
                      help='Batch size (default: 64)')
    args.add_argument("--model-path", type=str, default="facebook/opt-350m")
    args.add_argument("--model-base", type=str, default="facebook/opt-350m")
    args.add_argument("--max_length", type=int, default=77)
 
    config = args.parse_args()
    main(config)
