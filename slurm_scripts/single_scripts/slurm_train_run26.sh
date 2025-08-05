#!/bin/bash
#SBATCH --cpus-per-task=11
#SBATCH --error=/mnt/weka/slurm_logs/liliyu/img_edit_train/%j_%a_log.err
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/weka/slurm_logs/liliyu/img_edit_train/%j_%a_log.out
#SBATCH --signal=USR2@90
#SBATCH --wckey=submitit
#SBATCH --qos=hl
#SBATCH --job-name=all_statics

cd /home/liliyu/workspace/BAGEL
source .venv/bin/activate

# replace the variables with your own
# Fine-tuning
num_nodes=2
node_rank=0
master_addr=localhost
master_port=29500
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
resume_from=$model_path
# resume_from=results/pi_arxs_ur5_allview_seq_seedp1_gpu16_seq16384/checkpoints/0070000

GPUS=8

batch_size=1
seq_len=16384
export PYTHONPATH=/home/liliyu/workspace/BAGEL
total_gpus=$((num_nodes * GPUS))

# Fine-tuning
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29501  train/pretrain_unified_navit.py \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path $model_path \
  --resume-from $resume_from \
  --resume_model_only True \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens $seq_len \
  --max_num_tokens $seq_len \
  --max_num_tokens_per_sample $seq_len \
  --batch_size $batch_size \
  --dataset_config_file data/configs/seedp1_0.2_arxs_ur5_allview_endspan_nolast50.yaml  \
  --exp_name pi_arxs_ur5_allview_endspan_nolast50_seedp1_gpu${total_gpus}_seq${seq_len} \
  --wandb_runid 0 \
  --num_shard $total_gpus \
  --use_flex True \
  --visual_und False \
  --save_every 500
