#!/bin/bash

# Define default argument values
model_name="meta-llama/Meta-Llama-3.1-8B"
load_in_8bit="True"
dataset_name="data/custom_attack_log.csv"

# Run your command with the arguments
python model/script.py \
    --model_name=$model_name \
    --load_in_8bit=$load_in_8bit \
    --dataset_name=$dataset_name
