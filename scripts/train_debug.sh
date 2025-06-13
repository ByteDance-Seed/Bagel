# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# replace the variables with your own
# Fine-tuning
num_nodes=1
node_rank=0
master_addr=localhost
master_port=29500
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
resume_from=$model_path
resume_from=None
GPUS=4


batch_size=1
seq_len=10240
  # --resume-from $model_path \
# Fine-tuning
PYTHONPATH=. torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$GPUS \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/SEED_part1_ur5_endspan_448.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $resume_from \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens $seq_len \
  --max_num_tokens $seq_len \
  --max_num_tokens_per_sample $seq_len \
  --batch_size $batch_size \
  --wandb_name debugging \
  --wandb_runid 0 \
  --num_shard $GPUS \
  --use_flex True \
  --save_every 10

  # --visual_und False \
