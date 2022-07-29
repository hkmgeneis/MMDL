#!/bin/bash

current_time=$(date +"%Y_%m_%d_%H_%M_%S")
echo "$current_time"

images_dir="$1"
images_dir_split="$2"
prepare_type="$3"
images_dir_cn="$4"
labels_address="$5"
target_image_path="$6"


##step 1 prepare tumor ragion data
if false; then
image_suffix="*.svs"
python3.6 ../slide_to_tiles.py  \
--slide_image_root "$images_dir" \
--tiles_image_root "$images_dir_split" \
--size_square 512 \
--prepare_type "$prepare_type"
fi

## step 2 color normalization
if true; then
python3.6 ../stain_color_norm.py \
--source_input "$images_dir_split" \
--target_input "$target_image_path" \
--output_dir "$images_dir_cn" \
--file_types '*orig.png'
fi

## step 3 split data set
if false; then
python3.6 ../train_test_splitter.py \
--stained_tiles_home "$images_dir_cn" \
--label_dir_path "$labels_address"
fi

# end_time=$(data+"%Y_%m_%d_%H_%M_%S")
# echo "$end_time"
