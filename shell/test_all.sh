#!/bin/bash

# object paths
object_paths=(
    "foldingchair/102255/sequential_steps_5_full_48"
    "refrigerator/10905/sequential_steps_5_full_48"
    "laptop/10211/sequential_steps_5_full_48"
    "oven/101917/sequential_steps_5_full_48"
    "scissor/11100/sequential_steps_5_full_48"
    "stapler/103111/sequential_steps_5_full_48"
    "usb/100109/sequential_steps_5_full_48"
    "washer/103776/sequential_steps_5_full_48"
    "knife/101085/sequential_steps_5_full_48"
    "storage/45135/sequential_steps_5_full_48"
    "refrigerator/10489/random_steps_5_full_48"
    "storage/47254/random_steps_5_full_48"
    "storage/45503/random_steps_5_full_48"
)

# dataset root
dataset_roots=(
    "datasets/final"
    "datasets/partnet_mobility_blender"
)

# run shell script
for weight in 0.002; do
    for path in "${object_paths[@]}"; do
        for dataset_root in "${dataset_roots[@]}"; do
            python train.py -s "$dataset_root/$path" \
                --parsimony_weight_init $weight \
                --parsimony_weight_final $weight
        done
    done
done