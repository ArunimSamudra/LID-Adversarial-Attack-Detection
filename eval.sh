#!/bin/bash

# Define default argument values
model_name="meta-llama/Meta-Llama-3.1-8B"

# Run your command with the arguments
python eval/lid.py \
    --model_name=$model_name
