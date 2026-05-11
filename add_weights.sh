#!/bin/bash

# Define paths
surrage_path='/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT'
new_path="/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rb0123_15k"


# Copy checkpoint files
cp "$surrage_path/ae.safetensors" "$new_path/"
cp "$surrage_path/vit_config.json" "$new_path/"

cp "$surrage_path/llm_config.json" "$new_path/"


cp "$surrage_path/tokenizer.json" "$new_path/"

cp "$surrage_path/vocab.json" "$new_path/"
cp "$surrage_path/tokenizer_config.json" "$new_path/"

cp "$surrage_path/preprocessor_config.json" "$new_path/"
cp "$surrage_path/config.json" "$new_path/"
cp "$surrage_path/merges.txt" "$new_path/"