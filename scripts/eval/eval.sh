#!/bin/bash
CUDA_VISIBLE_DEVICES=1
INIT_MODEL_PATH="/hbox2dir"

coco_img_path="coco"
basename="fgclip2-base-patch16/"
# basename="fg-clipv2-base"

# S EVAL DOCCI-CN
# docci_cn_ann="docci/image_caption_trans.txt"
# python -m fgclip2.eval.long_trans_cn \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 196 \
#     --ann_file $docci_cn_ann \
#     --walk_type long \


# S EVAL DCI-CN
# dci_cn_ann="dci/image_caption_trans.txt"
# python -m fgclip2.eval.long_trans_cn \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 196 \
#     --ann_file $dci_cn_ann \
#     --walk_type long \


# S EVAL LIT-CN
# lit_cn_ann="LIT-CN/image_caption_long_cn.txt"
# python -m fgclip2.eval.long_trans_cn \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 196 \
#     --ann_file $lit_cn_ann \
#     --walk_type long \

# S EVAL Flickr-CNA
# flick_cna_ann="flickr30k_cna/flickr30k_cna_test.txt"
# python -m fgclip2.eval.flicker_cna \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \
#     --ann_file $flick_cna_ann \


# S EVAL COCO-CN
# coco_cn_ann="fgclip2/eval/pair.txt"
# python -m fgclip2.eval.coco_cn \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \
#     --ann_file $coco_cn_ann \
#     --image_folder $coco_img_path \


# S EVAL COCO-BOXCLS
# coco_box_ann="coco/annotations/instances_val2017.json"
# torchrun --master_port=8888 --nproc_per_node 8 -m fgclip2.eval.coco_box_ddp \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \
#     --ann_file $coco_box_ann \
#     --walk_type box \



# S EVAL LVIS-BOXCLS
# lvis_box_ann="lvis/lvis_v1_val.json"
# torchrun --master_port=8888 --nproc_per_node 8 -m fgclip2.eval.in1k.lvis_box_cls_ddp \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \
#     --ann_file $lvis_box_ann \
#     --walk_type box \


# S EVAL BoxClass-CN
# bcn_box_ann="BoxClass-CN/valid_category_data_total_zh.json"
# torchrun --master_port=8888 --nproc_per_node 8 -m fgclip2.eval.in1k.laion_cn_box_cls_ddp \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \
#     --ann_file $bcn_box_ann \
#     --walk_type box \
    

# S EVAL ShareGPT4V
# ShareGPT4V_ann="share-captioner_coco_lcs_sam_1246k_1107.json"
# python -m fgclip2.eval.share1k_retrieval \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 196 \
#     --ann_file $ShareGPT4V_ann \



# S EVAL DCI
# dci_ann="densely_captioned_images/annotations/"
# python -m fgclip2.eval.dci_retrieval \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 196 \
#     --ann_file $dci_ann \

# S EVAL COCO-RE
# coco_ann="coco/annotations/captions_val2017.json"
# python -m fgclip2.eval.coco_retrieval \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \
#     --ann_file $coco_ann \

# EVAL Flickr30k
# Flickr30k_ann="flickr30k/flickr30k_test.json"
# python -m fgclip2.eval.flickr30k_retrieval \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \
#     --ann_file $Flickr30k_ann \


# S EVAL IN-1K
# python -m fgclip2.eval.in1k.eval_IN1K \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \

# EVAL IN-v2
# python -m fgclip2.eval.inv2.inv2 \
#     --model-path $INIT_MODEL_PATH/$basename \
#     --model-base $INIT_MODEL_PATH/$basename \
#     --max_length 64 \