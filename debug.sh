#!/bin/bash
export PYTHONPATH="./:$PYTHONPATH"


# 1. 设置变量
MODEL_PATH="/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT"
MASTER_PORT=28544  # 你可以将这里改为 29500 或其他端口

# 2. 启动命令
# 注意：bash -c 后面现在使用的是双引号 " "
# 注意：内部的 "127.0.0.1" 变成了 \"127.0.0.1\"
wandb login 9e37f762624801dfc332b03f2ecefbf87153ed8f # chenming's wandb

srun -p efm_t --job-name=train_bagel_8gpu --gres=gpu:8 --time=1-00:00:00 --ntasks-per-node=1 -quotatype=reserved apptainer exec --nv --bind /mnt:/mnt /mnt/inspurfs/efm_t/zhuchenming/apptainer/bagel.sif \
    bash -c "torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --nproc_per_node=8 \
      --master_addr=\"127.0.0.1\" \
      --master_port=$MASTER_PORT \
      train/pretrain_unified_navit.py \
      --dataset_config_file ./data/configs/rule_base.yaml \
      --model_path $MODEL_PATH \
      --layer_module Qwen2MoTDecoderLayer \
      --max_latent_size 64 \
      --total_steps 30000 \
      --finetune_from_hf True \
      --auto_resume True \
      --resume-model-only False \
      --finetune-from-ema True \
      --log_every 1 \
      --results_dir /mnt/inspurfs/efm_t/zhuchenming/train_checkpoints/bagel/bagel_dl3dv/logs \
      --checkpoint_dir /mnt/inspurfs/efm_t/zhuchenming/train_checkpoints/bagel/bagel_dl3dv/ckpts \
      --lr 2e-5 \
      --visual_und False \
      --num_workers 1 \
      --save_every 5000 \
      --expected_num_tokens 13240 \
      --max_num_tokens 26520 \
      --max_num_tokens_per_sample 13240 \
      --wandb_offline True "