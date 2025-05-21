
name="openai/clip-vit-base-patch16"
# resume from openai please set from_openai==True

name="qihoo360/fg-clip-base"
# resume from fgclip please set from_openai==False


jsonfiles="FineHARD/jsonfiles"
image_root="data/grit-12m/"

deepspeed fgclip/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --base_model $name \
    --model_name_or_path $name \
    --data_path $jsonfiles \
    --image_folder $image_root \
    --output_dir ./checkpoints/fgclip-stage2 \
    --train_use_word_size 8 \
    --add_box_loss True \
    --use_hard_neg True \
    --from_openai True \
    --base_image_size 224 \
    --base_seq_length 77 \
    --max_seq_length 248 \
    --save_safetensors True \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 5000 \
    --save_total_limit 6 \
    --learning_rate 1e-6 \
    --weight_decay 0.001 \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "none" \
