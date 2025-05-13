#!/bin/bash
CUDA_VISIBLE_DEVICES=0

wholepath="qihoo360/fg-clip-base"
image_path="data/coco"

python -m myclip.eval.in_1K.coco_box_cls \
    --model-path $wholepath \
    --model-base $wholepath \
    --max_length 77 \
    --img_size 224 \
    --image-folder $image_path \
