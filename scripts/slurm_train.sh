#!/bin/bash
#SBATCH --cpus-per-task=11
#SBATCH --error=/mnt/weka/slurm_logs/liliyu/img_edit_train/%j_%a_log.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=img_edit_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/weka/slurm_logs/liliyu/img_edit_train/%j_%a_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=4320
#SBATCH --array=0-1
#SBATCH --wckey=submitit
#SBATCH --qos=high




# replace the variables with your own
# Fine-tuning
num_nodes=1
node_rank=0
master_addr=localhost
master_port=29500
resume_from=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
GPUS=8

export PYTHONPATH=/home/liliyu/workspace/BAGEL

# Fine-tuning
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29501  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path $resume_from \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --resume-from $resume_from \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --wandb_runid 1 \
  --num_shard $GPUS \
  --visual_und False



