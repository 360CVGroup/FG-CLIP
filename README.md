# FG-CLIP: Fine-Grained Visual and Textual Alignment

This repository is the official implementation of [FG-CLIP](https://github.com/360CVGroup/FG-CLIP).

**[FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/abs/2505.05071)** 
</br>
Chunyu Xie*, Bin Wang*, Fanjing Kong, Jincheng Li, Dawei Liang, Gengshen Zhang, Dawei Leng†, Yuhui Yin(*Equal Contribution, ✝Corresponding Author)
</br>
[![arXiv](https://img.shields.io/badge/arXiv-2505.05071-b31b1b.svg)](https://arxiv.org/abs/2505.05071)
[![ICML](https://img.shields.io/badge/ICML-2025-blue.svg)](https://icml.cc/Conferences/2025)

<!-- Contrastive Language-Image Pre-training (CLIP) excels in multimodal tasks such as image-text retrieval and zero-shot classification but struggles with fine-grained understanding due to its focus on coarse-grained short captions. To address this, we propose Fine-Grained CLIP (FG-CLIP), which enhances fine-grained understanding through three key innovations. First, we leverage large multimodal models to generate 1.6 billion long caption-image pairs for capturing global-level semantic details. Second, a high-quality dataset is constructed with 12 million images and 40 million region-specific bounding boxes aligned with detailed captions to ensure precise, context-rich representations. Third, 10 million hard fine-grained negative samples are incorporated to improve the model's ability to distinguish subtle semantic differences. Corresponding training methods are meticulously designed for these data. Extensive experiments demonstrate that FG-CLIP outperforms the original CLIP and other state-of-the-art methods across various downstream tasks, including fine-grained understanding, open-vocabulary object detection, image-text retrieval, and general multimodal benchmarks. These results highlight FG-CLIP's effectiveness in capturing fine-grained image details and improving overall model performance. -->


<!-- ## Model Performance -->
<!-- ### Long/short caption image-text retrieval, and zero-shot image classification..  -->

 <p align="center">
  <img src="./use_imgs/radar_chart_methods.png"  width="500" height="440"/>
</p>


## 🔥 News
- 🚀 **[2025/05/13]** We have uploaded the dataset FgGRN-12M (Fine-Grained visual Grounding+Recaption+Negative) to 🤗(https://huggingface.co/datasets/qihoo360/FgGRN-12M)!
- 🚀 **[2025/05/09]** We have uploaded the model to 🤗(https://huggingface.co/qihoo360/fg-clip-large), which supports quick and easy usage!
- 🚀 **[2025/05/09]** We have updated the FG-CLIP github repository, and now you can test our models!
- 🚀 **[2025/05/09]** We released the paper of [FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/abs/2505.05071).
- 🚀 **[2025/05/02]** FG-CLIP has been accepted by ICML'25.

## Contents
- [Model Framework](#ModelFramework)
- [Install](#install)
- [Model Zoo](#modelzoo)
- [Quick Start](#quickstart)
- [Train](#train)
- [Evaluation](#evaluation)



## Model Framework
FG-CLIP’s training proceeds in two stages: the first stage leverages
global-level caption-image pairs to achieve initial fine-grained alignment, while the second stage supplements these with additional
region-level captions, including detailed region captions and positive/negative region descriptions to further refine the alignment.
<p align="center">
  <img src="./use_imgs/fgclip_strc.png" width=80%/>

## Install

```Shell
conda create -n FGCLIP python=3.10 -y
conda activate FGCLIP
cd FG-CLIP && pip install -e .
```
## Model Zoo: 
[🤗Vit-B@224px](https://huggingface.co/qihoo360/fg-clip-base)
</br>
[🤗Vit-L@336px](https://huggingface.co/qihoo360/fg-clip-large)

<!-- <p align="center">
  <img src="./use_imgs/fgclip_strc.png" width=80%/>
</p> -->

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


model_root = "qihoo360/fg-clip-base"
image_size=224
model = AutoModelForCausalLM.from_pretrained(model_root,trust_remote_code=True).cuda()

device = model.device

tokenizer = AutoTokenizer.from_pretrained(model_root)
image_processor = AutoImageProcessor.from_pretrained(model_root)
```


### Retrieval

```Shell

img_root = "FG-CLIP/use_imgs/cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")
image = image.resize((image_size,image_size))

image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

# NOTE Short captions: max_length=77 && walk_short_pos=True
walk_short_pos = True
captions=["a photo of a cat", "a photo of a dog"]
caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)

# NOTE Long captions: max_length=248 && walk_short_pos=False
# ......

with torch.no_grad():
  image_feature = model.get_image_features(image_input)
  text_feature = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)
  image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
  text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

logits_per_image = image_feature @ text_feature.T 
probs = logits_per_image.softmax(dim=1) 
print(probs)

```

### Dense feature effect display

```Shell

import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


img_root = "FG-CLIP/use_imgs/cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")
image = image.resize((image_size,image_size))

image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

with torch.no_grad():
    dense_image_feature = model.get_image_dense_features(image_input)
    captions = ["white cat"]
    caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
    text_feature = model.get_text_features(caption_input,walk_short_pos=True)
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
similarity = similarity.cpu().numpy()
patch_size = int(math.sqrt(similarity.shape[0]))


original_shape = (patch_size, patch_size)
show_image = similarity.reshape(original_shape) 


plt.figure(figsize=(6, 6))
plt.imshow(show_image)
plt.title('similarity Visualization')
plt.axis('off')  
plt.savefig("FG-CLIP/use_imgs/FGCLIP_dfcolor_cat.png")

```
<!-- /home/jovyan/wangbin-home-shcdt/image_text_match/FG-CLIP/use_imgs/FGCLIP_dfcolor_cat.png -->
 <p align="left">
  <img src="./use_imgs/FGCLIP_dfcolor_cat.png" width=25%/>
</p>



## Train

### Data Preparation
 [[Data: FgGRN-12M](data/data.md)]

Our training and inference code is completely based on the transformers repository provided by huggingface, which is a very easy to use and easy to reproduce. We have provided the training script in the scripts directory.
</br>
[🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers)
</br>

Our training script supports the use of zero2, tf32 acceleration, and bf16 precision (note that fp16 precision may cause gradient NAN). If you do not meet the above conditions, please turn off tf32 and replace deepspeed startup with torchrun.
</br>
```Shell
bash scripts/train.sh
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

```none
FG-CLIP
├── ...
├── data
|   ├── share-captioner_coco_lcs_sam_1246k_1107.json
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── annotations
│   │   |   ├── captions_val2017.json
│   ├── fgovd
│   │   ├── h_attributes_llava.jsonl
│   │   ├── m_attributes_llava.jsonl
│   │   ├── e_attributes_llava.jsonl
│   │   ├── shuffle_negatives_llava.jsonl
│   ├── sam
│   │   ├── images
│   │   |   ├──138
│   ├── densely_captioned_images
│   │   ├── annotations
│   │   |   ├── ...
│   ├── IN1K_val
│   │   ├── val
│   │   ├── imagenet2012_mapclsloc.txt
│   ├── imagenetv2-matched-frequency-format-val
├── ...
```



### Long/Short caption image-text retrieval

```Shell
bash scripts/eval/eval_long.sh
bash scripts/eval/eval_coco_retrieval.sh
```
### Bounding box classification of COCO

```Shell
bash scripts/eval/eval_coco_boxcls.sh
```

### FG-OVD benchmark
```Shell
bash scripts/eval/eval_fgovd.sh
```

### Zero-Shot classification
```Shell
bash scripts/eval/eval_in1k.sh
bash scripts/eval/eval_v2.sh
```




<!-- ## Acknowledgement -->
## We Are Hiring
We are seeking academic interns in the Multimodal field. If interested, please send your resume to xiechunyu@360.cn.
## Citation
If you find FG-CLIP useful for your research and applications, please cite using this BibTeX:

```
@article{xie2025fgclip,
      title={FG-CLIP: Fine-Grained Visual and Textual Alignment}, 
      author={Chunyu Xie and Bin Wang and Fanjing Kong and Jincheng Li and Dawei Liang and Gengshen Zhang and Dawei Leng and Yuhui Yin},
      year={2025},
      eprint={2505.05071},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.05071}, 
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