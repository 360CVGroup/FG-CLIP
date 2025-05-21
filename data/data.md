<!-- ---
tags:
- Clip
- Grounding
- Caption
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: zero-shot-image-classification
--- -->
# FG-CLIP: Fine-Grained Visual and Textual Alignment
**[FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/abs/2505.05071)** 
</br>
Chunyu Xie*, Bin Wang*, Fanjing Kong, Jincheng Li, Dawei Liang, Gengshen Zhang, Dawei Leng†, Yuhui Yin(*Equal Contribution, ✝Corresponding Author)
</br>
[![arXiv](https://img.shields.io/badge/arXiv-2505.05071-b31b1b.svg)](https://arxiv.org/abs/2505.05071)
[![ICML](https://img.shields.io/badge/ICML-2025-blue.svg)](https://icml.cc/Conferences/2025)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/360CVGroup/FG-CLIP)
 <p align="center">
  <img src="https://huggingface.co/qihoo360/fg-clip-large/resolve/main/radar_chart_methods.png"  width="500" height="440"/>
</p>

## Model Framework
FG-CLIP’s training proceeds in two stages: the first stage leverages
global-level caption-image pairs to achieve initial fine-grained alignment, while the second stage supplements these with additional
region-level captions, including detailed region captions and positive/negative region descriptions to further refine the alignment.
<p align="center">
  <img src="https://huggingface.co/qihoo360/fg-clip-large/resolve/main/fgclip_strc.png" width=80%/>
</p>



# Data Preparation
To run the training code for FG-CLIP, please follow the following step.

### Step 1: Download the model

Download the FG-CLIP model from this link. [🤗Vit-L@336px](https://huggingface.co/qihoo360/fg-clip-large) or
Download the OpenAI CLIP model from this link. [🤗Vit-L@336px](https://huggingface.co/openai/clip-vit-large-patch14-336)


### Step 2: Prepare FineHARD (Fine-Grained Visual Grounding+Recaption+Hard Negative Dataset) Dataset

First, pull the dataset from the following link.
[🤗FineHARD](https://huggingface.co/datasets/qihoo360/FineHARD)，After downloading, unzip all compressed files, you will obtain the following file structure:



```none
FineHARD
├── url2key_jsons
|   ├── url2key_coyo_image_0.json
|   ├── ...
│   ├── url2key_coyo_image_20.json
├── jsonfiles
|   ├── 2024-12-06_18-32-53_results_10_218_126_44_1025.json
│   ├── 2024-12-06_18-33-17_results_llama70b-shcdt-h100-4gpus-no-2.json
│   ├──...
├── coyo_image_0
|   ├── 00000.parquet
│   ├── 00001.parquet
│   ├── ...
│   ├── 00099.parquet
├── coyo_image_1
|   ├── 00000.parquet
│   ├── 00001.parquet
│   ├── ...
│   ├── 00099.parquet
├── ...
├── coyo_image_20
|   ├── 00000.parquet
│   ├── 00001.parquet
│   ├── ...
│   ├── 00050.parquet
├── ...
```

Subsequently, you need to install the `img2dataset` package. You can do this by running the following command:

```bash
pip install img2dataset
```

Set the `file_in` parameter in the script (`data/get_data.sh`) according to the download path of the data, and also set the directory where you expect to save the files (`pre_dir`, `dir_save`). Subsequently, execute the following commands.


```bash
bash data/get_data.sh
```

Due to the randomness in downloading, the image names corresponding to the URLs do not match the names of the images we are using. Therefore, a conversion is needed. This step requires using the `url2key_jsons/*.json` file included in the FineHARD dataset. Also, you can use the files in `url2key_jsons/*.json` to check the download links of all the images we used.

```bash
python -m data.convert_image_name \
    --url2key_json FineHARD/url2key_jsons \
    --down_file_root data/down-grit-12m/ \
    --num_parent_folders 21 \
    --num_subfolders_per_parent 100 \
    --resave_file_root data/grit-12m/ \

rm -r data/down-grit-12m/
```

```none
FG-CLIP
├── ...
├── FineHARD
|   ├── jsonfiles
|   |   ├── 2024-12-06_18-32-53_results_10_218_126_44_1025.json
|   |   ├── 2024-12-06_18-33-17_results_llama70b-shcdt-h100-4gpus-no-2.json
|   |   ├──...
|   ├── ...
├── data
|   ├── grit-12m
|   |   ├── coyo_image_0
|   |   |   ├──00000
|   |   |   ├──00001
|   |   |   ├──...
|   |   |   ├──00099
|   |   ├── coyo_image_1
|   |   |   ├──00000
|   |   |   ├──00001
|   |   |   ├──...
|   |   |   ├──00099
|   |   ├── ...
|   |   ├── coyo_image_20
|   |   |   ├──00000
|   |   |   ├──00001
|   |   |   ├──...
|   |   |   ├──00050
├── ...
```