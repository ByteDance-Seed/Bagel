# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export OPENAI_API_KEY=$openai_api_key

GPUS=8

export CUDA_VISIBLE_DEVICES=

# ============ Generate Images ============
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_unicbench.py \
    --output_dir $output_path/bagel \
    --max_latent_size 64 \
    --model-path $model_path \
    --language en \
    --dataset_name xiaotanhua/UnicBench

# ============ Evaluate ============
cd ./eval/gen/unicbench
python eval_pipeline.py \
    --data_path xiaotanhua/UnicBench \
    --save_dir /path/to/results \
    --edit_model_name bagel \
    --vlm_model_name gpt-4.1 \
    --languages en \
    --num_workers 8