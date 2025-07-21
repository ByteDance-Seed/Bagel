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
    ./eval/gen/gen_images_mp_wise.py \
    --output_dir $output_path/images \
    --metadata-file ./eval/gen/wise/final_data.json \
    --resolution 1024 \
    --max-latent_size 64 \
    --model-path $model_path \
    --batch_size $BATCH_SIZE \
    --think


# calculate score
python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/cultural_common_sense.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/spatio-temporal_reasoning.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/natural_science.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/cal_score.py \
        --output_dir $output_path