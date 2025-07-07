# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# replace the variables with your own
# Fine-tuning
num_nodes=8
node_rank=0
master_addr=localhost
master_port=29500
# model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
  # --resume-from $model_path \
# resume_from=$model_path
resume_from=None
GPUS=1


batch_size=1
seq_len=10240
  # --resume-from $resume_from \
# Fine-tuning
PYTHONPATH=. torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$GPUS \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/seedp1_0.2_h1g1_allview_endspan.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --exp-checkpoint-dir /mnt/weka/checkpoints/liliyu/bagel \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens $seq_len \
  --max_num_tokens $seq_len \
  --max_num_tokens_per_sample $seq_len \
  --batch_size $batch_size \
  --exp_name debug_allview_video7 \
  --wandb_runid 4 \
  --num_shard $GPUS \
  --visual_und False \
  --save_every 10

