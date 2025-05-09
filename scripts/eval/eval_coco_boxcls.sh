#!/bin/bash
CUDA_VISIBLE_DEVICES=0

basename="openai/clip-vit-base-patch16"


wholepath="FGCLIP-VITB-Stage2"
image_path="/mm-datasets/public/coco"

python -m myclip.eval.in_1K.coco_box_cls \
    --model-path $wholepath \
    --model-base $INIT_MODEL_PATH/$basename \
    --max_length 77 \
    --img_size 224 \
    --image-folder $image_path \
