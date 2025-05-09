#!/bin/bash
CUDA_VISIBLE_DEVICES=0

wholepath="qihoo360/fg-clip-base"
image_path="data/coco"

python -m fgclip.eval.coco_retrieval \
    --model-path $wholepath \
    --model-base $wholepath \
    --max_length 77 \
    --image_size 224 \
    --image-folder $image_path \
