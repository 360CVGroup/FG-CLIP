[**中文说明**](README.md) | [**English**](README_en.md)

# FG-CLIP 2: A Bilingual Fine-grained Vision-language Alignment Model

This repository is the official implementation of FG-CLIP and FG-CLIP 2, a new generation of text-image cross-modal model excels in fine-grained discrimination and embedding.

FG-CLIP 2 is the foundation model for fine-grained vision-language understanding in both English and Chinese. 
Across 29 datasets and 8 diverse tasks, it consistently surpasses recent strong baselines such as SigLIP 2 and MetaCLIP 2, achieving the best reported performance to date in both languages. 

**[FG-CLIP 2: A Bilingual Fine-grained Vision-language Alignment Model](https://arxiv.org/abs/2510.10921)** 
</br>
Chunyu Xie*, Bin Wang*, Fanjing Kong, Jincheng Li, Dawei Liang, Ji Ao, Dawei Leng†, Yuhui Yin (*Equal Contribution, ✝Corresponding Author)
</br>
[![arXiv](https://img.shields.io/badge/arXiv-2510.10921-b31b1b.svg)](https://arxiv.org/abs/2510.10921)
[![HF-model](https://img.shields.io/badge/Model-Collection🤗-yellow.svg)](https://huggingface.co/collections/qihoo360/fg-clip-2-68ecbf9c548623bb78bc7913)
[![HF-data](https://img.shields.io/badge/Benchmark-Collection🤗-yellow.svg)](https://huggingface.co/collections/qihoo360/fg-clip-2-68ecbf9c548623bb78bc7913)
[![API+MCP](https://img.shields.io/badge/API/MCP-FG--CLIPv2-green.svg)](https://research.360.cn/sass/index)

**[FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/abs/2505.05071)** ([code branch: v1.0](https://github.com/360CVGroup/FG-CLIP/tree/v1.0))
</br>
Chunyu Xie*, Bin Wang*, Fanjing Kong, Jincheng Li, Dawei Liang, Gengshen Zhang, Dawei Leng†, Yuhui Yin (*Equal Contribution, ✝Corresponding Author)
</br>
[![arXiv](https://img.shields.io/badge/arXiv-2505.05071-b31b1b.svg)](https://arxiv.org/abs/2505.05071)
[![ICML](https://img.shields.io/badge/ICML-2025-blue.svg)](https://icml.cc/Conferences/2025)
[![HF-model](https://img.shields.io/badge/Model-Collection🤗-yellow.svg)](https://huggingface.co/collections/qihoo360/fg-clip-681da45d4acfb65c240a6d08)
[![HF-data](https://img.shields.io/badge/Data-FineHARD🤗-yellow.svg)](https://huggingface.co/datasets/qihoo360/FineHARD)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-FG--CLIP-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/360CVGroup/FG-CLIP)

 <p align="center">
  <img src="./use_imgs/FGCLIP2_compare_all_n.png"  width="500" height="440"/>
</p>

## 🔥 News
- 🚀 **[2025/10/14]** We have uploaded the FG-CLIP 2 code and models.
- 🚀 **[2025/10/14]** We released the paper of [FG-CLIP 2: A Bilingual Fine-grained Vision-language Alignment Model](https://arxiv.org/abs/2510.10921).
- 🚀 **[2025/09/29]** We just open-sourced MCP server implementation for FG-CLIP, check [FGCLIP-MCP](https://github.com/360CVGroup/FGCLIP-MCP) for details.
- 🚀 **[2025/07/29]** We provide API access of FG-CLIP 2 base model, which out-performs FG-CLIP by significant margin, check [research.360.cn](https://research.360.cn/sass/index) for details.
- 🚀 **[2025/07/09]** We created two new demos for easy testing, for [fine-grained retrieval](https://huggingface.co/spaces/qihoo360/FG-CLIP-Retrieval-demo) and [dense feature display](https://huggingface.co/spaces/qihoo360/FG-CLIP-Densefeature-demo).
- 🚀 **[2025/05/09]** We have uploaded the model to 🤗(https://huggingface.co/qihoo360/fg-clip-large), which supports quick and easy usage!
- 🚀 **[2025/05/09]** We have updated the FG-CLIP github repository, and now you can test our models!
- 🚀 **[2025/05/09]** We released the paper of [FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/abs/2505.05071).
- 🚀 **[2025/05/02]** FG-CLIP has been accepted by ICML'25.


<!-- ## Overview




Fine-grained vision-language understanding requires precise alignment between visual content and linguistic descriptions, a capability that remains limited in current models, particularly in non-English settings. While models like CLIP perform well on global alignment, they often struggle to capture fine-grained details in object attributes, spatial relations, and linguistic expressions, with limited support for bilingual comprehension. To address these challenges, we introduce FG-CLIP 2, a bilingual vision-language model designed to advance fine-grained alignment for both English and Chinese. The key ingredients of FG-CLIP 2 are summarized below.

- Rich Fine-Grained Supervision. Including region-text matching and long-caption modeling, alongside multiple discriminative objectives. We further introduce the Textual Intra-modal Contrastive (TIC) loss to better distinguish semantically similar captions.
- Bilingual Multimodal Data. Trained on a carefully curated mixture of large-scale English and Chinese data, FG-CLIP 2 achieves powerful bilingual performance.
- Performance. Extensive experiments on 29 datasets across 8 tasks show that FG-CLIP 2 outperforms existing methods, achieving state-of-the-art results in both languages.
- Chinese Multimodal Benchmark. To enable rigorous evaluation, we present a new benchmark for Chinese multimodal understanding, featuring long-caption retrieval and bounding box classification. -->



<!-- ## Model Performance -->
<!-- ### Long/short caption image-text retrieval, and zero-shot image classification..  -->



## Contents
- [Model Framework](#model-framework)
- [Install](#install)
- [Model Zoo](#model-zoo)
- [Quick Start](#quick-start)
- [Train](#train)
- [Evaluation](#evaluation)



## Model Framework
Our approach employs a two-stage hierarchical learning framework that progressively enhances vision-language alignment from global semantics to fine-grained details.

Stage 1: Global Semantic Alignment. We begin with large-scale image-text pairs, each annotated with both a short caption (for concise scene-level description) and a long caption (for rich contextual detail). Training on this bilingual corpus enables strong global alignment, establishing a robust foundation for cross-modal understanding in both English and Chinese.

Stage 2: Fine-Grained Visual-Language Learning. Building upon the globally aligned representation, we add region-level supervision and multiple fine-grained objectives to sharpen local correspondences. Specifically, this stage incorporates:

- Fine-Grained Visual Learning: region–caption alignment via RoIAlign-extracted region features and phrase-level descriptions.
- Fine-Grained Textual Learning: discrimination of subtle textual differences using hard negatives with perturbed attributes.
- Cross-Modal Rank Loss with Global Threshold Synchronization: dynamic margin-based ranking with globally synchronized thresholds for stable hard negative mining.
- Textual Intra-modal Contrastive Loss: intra-language contrastive learning to separate semantically similar but distinct region captions.
<p align="center">
  <img src="./use_imgs/framework.png" width=80%/>

## Install

```Shell
conda create -n FGCLIP2 python=3.10 -y
conda activate FGCLIP2
cd FG-CLIP && pip install -e .
```
## Model Zoo

|Models |           ViT           |                       Model Weights                      |                           Demo                           |
|:-----------|:-----------------------:|:---------------------------------------------------------:|:--------------------------------------------------------:|
| FG-CLIP-Base   | vit-base-patch16-224 | [🤗Huggingface](https://huggingface.co/qihoo360/fg-clip-base)  | [Retrieval](https://huggingface.co/spaces/qihoo360/FG-CLIP-Retrieval-demo) & [Dense Feature](https://huggingface.co/spaces/qihoo360/FG-CLIP-Densefeature-demo) |
|  FG-CLIP-Large   | vit-large-patch14-336 | 🤗[Huggingface](https://huggingface.co/qihoo360/fg-clip-large)  |  |
| FG-CLIP2-Base   | vit-base-patch16 | [🤗Huggingface](https://huggingface.co/qihoo360/fg-clip2-base)  | [Retrieval](https://huggingface.co/spaces/qihoo360/FG-CLIP2-Retrieval-demo) & [Dense Feature](https://huggingface.co/spaces/qihoo360/FG-CLIP2-Densefeature-demo) |
|  FG-CLIP2-Large   | vit-large-patch16 | [🤗Huggingface](https://huggingface.co/qihoo360/fg-clip2-large)  |  |
|  FG-CLIP2-So400m   | vit-so400m-patch16 | [🤗Huggingface](https://huggingface.co/qihoo360/fg-clip2-so400m)  |  |

## Benchmark

|Datasets |          Link          | 
|:-----------|:-----------------------:|
| LIT-CN   | [🤗https://huggingface.co/datasets/qihoo360/LIT-CN](https://huggingface.co/datasets/qihoo360/LIT-CN)  | 
|  DCI-CN   |  🤗[https://huggingface.co/datasets/qihoo360/DCI-CN](https://huggingface.co/datasets/qihoo360/DCI-CN)  | 
| DOCCI-CN   |  [🤗https://huggingface.co/datasets/qihoo360/DOCCI-CN](https://huggingface.co/datasets/qihoo360/DOCCI-CN)  |
|  BoxClass-CN   |  [🤗https://huggingface.co/datasets/qihoo360/BoxClass-CN](https://huggingface.co/datasets/qihoo360/BoxClass-CN)  | 

## Quick Start 🤗

### Load Model
```Shell
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


model_root = "fgclip2-base-patch16"
model = AutoModelForCausalLM.from_pretrained(model_root,trust_remote_code=True).cuda()

device = model.device

tokenizer = AutoTokenizer.from_pretrained(model_root)
image_processor = AutoImageProcessor.from_pretrained(model_root)

```


### Retrieval

```Shell
def determine_max_value(image):
    w,h = image.size
    max_val = (w//16)*(h//16)
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

img_root = "cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")

image_input = image_processor(images=image, max_num_patches=determine_max_value(image), return_tensors="pt").to(device)

# NOTE Short captions: max_length=64 walk_type="short"(default)
# NOTE Long captions: max_length=196 walk_type="long"

captions = [
"A minimalist-style bedroom corner with a black metal clothing rack holding several beige and white garments, two pairs of light-colored shoes on the shelf below, a potted green plant nearby, and to the left, a bed made with white sheets and gray pillows.",
"A minimalist-style bedroom corner with a black metal clothing rack holding several red and blue garments, two pairs of black high heels on the shelf below, a potted green plant nearby, and to the left, a bed made with white sheets and gray pillows.",
"A minimalist-style bedroom corner with a black metal clothing rack holding several beige and white garments, two pairs of sneakers on the shelf below, a potted cactus nearby, and to the left, a bed made with white sheets and gray pillows.",
"A bustling street market with fruit-filled stalls, skyscrapers in the background, and people shopping amid the noise and activity."
]
captions = [caption.lower() for caption in captions]

caption_input = tokenizer(captions, padding="max_length", max_length=196, truncation=True, return_tensors="pt").to(device)


with torch.no_grad():
  image_feature = model.get_image_features(**image_input)
  text_feature = model.get_text_features(**caption_input)
  image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
  text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

logits_per_image = image_feature @ text_feature.T
logit_scale, logit_bias = model.logit_scale.to(text_feature.device), model.logit_bias.to(text_feature.device)
logits_per_image = logits_per_image * logit_scale.exp() + logit_bias

```
 <p align="left">
  <img src="use_imgs\en_re_demo.png" width=100%/>
</p>

### Dense feature effect display

```Shell

import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def resize_short_edge(image, target_size=2048):
    if isinstance(image, str):
        image = Image.open(image)
    width, height = image.size
    short_edge = min(width, height)

    if short_edge >= target_size:
        return image
    scale = target_size / short_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = image.resize((new_width, new_height))
    return resized_image

img_root = "cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")
image = resize_short_edge(image,target_size=2048)

image_input = image_processor(images=image, max_num_patches=16384, return_tensors="pt").to(device)
captions = ["电脑","黑猫","窗户","window","white cat","book"]

with torch.no_grad():
    dense_image_feature = model.get_image_dense_feature(**image_input)
    
    spatial_values = image_input["spatial_shapes"][0]
    real_h = spatial_values[0].item()
    real_w = spatial_values[1].item()
    real_pixel_tokens_num = real_w*real_h
    dense_image_feature = dense_image_feature[0][:real_pixel_tokens_num]
    captions = [caption.lower() for caption in captions]
    caption_input = tokenizer(captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)

    text_feature = model.get_text_features(**caption_input, walk_type="box")
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

similarity = dense_image_feature @ text_feature.T
similarity = similarity.cpu()


num_classes = len(captions)
cols = 3
rows = (num_classes + cols - 1) // cols


aspect_ratio = real_w / real_h 

fig_width_inch = 3 * cols        
fig_height_inch = fig_width_inch / aspect_ratio * rows / cols  

fig, axes = plt.subplots(rows, cols, figsize=(fig_width_inch, fig_height_inch))
fig.subplots_adjust(wspace=0.01, hspace=0.01)

if num_classes == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for cls_index in range(num_classes):
    similarity_map = similarity[:, cls_index].cpu().numpy()
    show_image = similarity_map.reshape((real_h, real_w))

    ax = axes[cls_index]
    ax.imshow(show_image, cmap='viridis', aspect='equal')  
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')


for idx in range(num_classes, len(axes)):
    axes[idx].axis('off')

savename = "FGCLIP2_dfcolor_cat_all_2K.png"
plt.savefig(savename, dpi=150, bbox_inches='tight', pad_inches=0.05)
plt.close()
```

 <p align="left">
  <img src="use_imgs\FGCLIP2_dfcolor_cat_all_2K.png" width=100%/>
</p>

## Train

### Data Preparation

We provide code for the second stage training using the [🤗FineHARD dataset](https://huggingface.co/datasets/qihoo360/FineHARD). FineHARD dataset includes 12 million images, 40 million bounding boxes with fine-grained region descriptions, and 10 million hard negative samples.
</br>
For data preparation, please refer to [Data: FineHARD](data/data.md)


### Ready for Training
Our training and inference code is completely based on the transformers repository provided by huggingface, which is a very easy to use and easy to reproduce. We have provided the training script in the scripts directory.
</br>
[🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers)
</br>
Our training script supports the use of zero2, tf32 acceleration, and bf16 precision (note that fp16 precision may cause gradient NAN). If you do not meet the above conditions, please turn off tf32 and replace deepspeed startup with torchrun.
</br>
```Shell
bash scripts/train/stage2_fgclip2.sh
```


## Evaluation
### Data Preparation
Download the share-captioner_coco_lcs_sam_1246k_1107.json from the following link 
https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json

Download the CocoCaptions from the following link nd put them into data/coco/annotations/
https://github.com/tylin/coco-caption

Download the COCO from the following link and put them into data/coco
https://cocodataset.org/dataset

Captions of DCI are from the following links and put them into data/densely_captioned_images
https://github.com/facebookresearch/DCI

ImageNet-1K from from the following links and put them into data/IN1K_val
https://image-net.org/

ImageNet-v2 from the following links and put them into data/imagenetv2-matched-frequency-format-val
https://opendatalab.com/OpenDataLab/ImageNetV2/tree/main


```bash
bash scripts/eval/eval.sh
```




<!-- ## Acknowledgement -->
## We Are Hiring
We are seeking academic interns in the Multimodal field. If interested, please send your resume to xiechunyu@360.cn.
## Citation
If you find FG-CLIP 2 useful for your research and applications, please cite using this BibTeX:

```
@article{xie2025fg2,
  title={FG-CLIP 2: A Bilingual Fine-grained Vision-language Alignment Model},
  author={Xie, Chunyu and Wang, Bin and Kong, Fanjing and Li, Jincheng and Liang, Dawei and Ao, Ji and Leng, Dawei and Yin, Yuhui},
  journal={arXiv preprint arXiv:2510.10921},
  year={2025}
}
```
```
@article{xie2025fg,
  title={FG-CLIP: Fine-Grained Visual and Textual Alignment},
  author={Xie, Chunyu and Wang, Bin and Kong, Fanjing and Li, Jincheng and Liang, Dawei and Zhang, Gengshen and Leng, Dawei and Yin, Yuhui},
  journal={arXiv preprint arXiv:2505.05071},
  year={2025}
}
```





## License

This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.
The content of this project itself is licensed under the [Apache license 2.0](./LICENSE).

## Related Projects
This work wouldn't be possible without the incredible open-source code of these projects. Huge thanks!
- [CLIPSelf](https://github.com/wusize/CLIPSelf.git)
- [FineCLIP](https://github.com/Timsty1/FineCLIP)
- [LLava](https://github.com/haotian-liu/LLaVA)
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP.git)
