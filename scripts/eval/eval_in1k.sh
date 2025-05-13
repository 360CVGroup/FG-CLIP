INIT_MODEL_PATH="/hbox2dir"

wholepath="qihoo360/fg-clip-base"

image_folder="data/IN1K_val/val"
map_idx_file="data/IN1K_val/imagenet2012_mapclsloc.txt"

python -m fgclip.eval.in_1K.eval_in1k \
    --model-path $wholepath \
    --model-base $wholepath \
    --max_length 77 \
    --image_size 224 \
    --image_folder $image_folder \
    --map_idx_file $map_idx_file \
    --b 64 \
