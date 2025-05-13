
wholepath="qihoo360/fg-clip-base"

data_dir="data/imagenetv2-matched-frequency-format-val"


python -m fgclip.eval.in_v2.eval_inv2 \
    --model-path $wholepath \
    --model-base $wholepath \
    --max_length 77 \
    --image_size 224 \
    --d $data_dir \
    --b 64 \
