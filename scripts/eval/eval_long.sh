#!/bin/bash
CUDA_VISIBLE_DEVICES=0


wholepath="qihoo360/fg-clip-base"


share_1k_jsonfilepath="share-captioner_coco_lcs_sam_1246k_1107.json"
share_1k_image_path="data/coco"
python -m fgclip.eval.shareg4v_retrieval \
    --model-path $wholepath \
    --model-base $wholepath \
    --max_length 248 \
    --image_size 224 \
    --jsonfile_path $share_1k_jsonfilepath \
    --image-folder $share_1k_image_path \
    

dci_jsonfilepath="data/densely_captioned_images/annotations"
dci_image_path="data/sam/images/138/"

python -m fgclip.eval.dci_retrieval \
    --model-path $wholepath \
    --model-base $wholepath \
    --max_length 248 \
    --image_size 224 \
    --jsonfile_path $dci_jsonfilepath \
    --image-folder $dci_image_path \

