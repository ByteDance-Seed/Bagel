# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export OPENAI_API_KEY=$openai_api_key

GPUS=8


# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_rise.py \
    --output_dir $output_path/bagel \
    --metadata_file ./eval/gen/rise/data/datav2_total_w_subtask.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think


# calculate score
python ./eval/gen/rise/gpt_eval.py \
    --data ./eval/gen/rise/data/datav2_total_w_subtask.json \
    --input ./eval/gen/rise/data \
    --output $output_path/bagel