# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# replace the variables with your own
# Fine-tuning
num_nodes=1
node_rank=0
master_addr=localhost
master_port=29500
resume_from=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
# resume_from=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
GPUS=8


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
  --dataset_config_file ./data/configs/SEED_part2_arx_50step.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path $resume_from \
  --max_latent_size 64 \
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
  --wandb_name testtt \
  --wandb_runid 5 \
  --num_shard $GPUS \
  --visual_und False \
  --use_flex True \
  --save_every 10

