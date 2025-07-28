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
#SBATCH --job-name=bagel
#SBATCH --qos=hl

# Check if config name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_name>"
    echo "Example: $0 seedp1_0.2_arx_biarm_allview_endspan"
    exit 1
fi


# Get config name from command line argument
config_name=$1
echo "Config name: $config_name"
# Rename the job to use the config namegf
scontrol update job $SLURM_JOB_ID name=bagel_$config_name

# Get config name from command line argument
post_fix="${2:-_}"


cd /home/liliyu/workspace/BAGEL
source .venv/bin/activate

# replace the variables with your own
# Fine-tuning
num_nodes=$SLURM_NNODES
node_rank=$SLURM_NODEID

master_addr=localhost
master_port=29503
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
# resume_from=/mnt/weka/checkpoints/liliyu/bagel_ckpt/seed_blip3o_all_robots_jul19_gpu64seq16384_pretrain/checkpoints/0008500
resume_from=$model_path
ckpt_dir=/mnt/weka/checkpoints/liliyu/bagel_ckpt/
GPUS=8

batch_size=1
seq_len=16384
export PYTHONPATH=/home/liliyu/workspace/BAGEL
total_gpus=$((num_nodes * GPUS))
num_shard=8
num_replicate=$((total_gpus/num_shard))

timestep_shift=1.0

# Fine-tuning
srun torchrun --nnodes=$num_nodes --nproc_per_node=$GPUS \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:$master_port  train/pretrain_unified_navit.py \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path $model_path \
  --resume-from $resume_from \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --exp_checkpoint_dir $ckpt_dir \
  --checkpoint_dir $ckpt_dir \
  --log_every 10 \
  --lr 2e-5 \
  --num_worker 1 \
  --timestep_shift $timestep_shift \
  --expected_num_tokens $seq_len \
  --max_num_tokens $seq_len \
  --max_num_tokens_per_sample $seq_len \
  --batch_size $batch_size \
  --dataset_config_file data/configs/${config_name}.yaml  \
  --exp_name ${config_name}_t${timestep_shift}_gpu${total_gpus}_seq${seq_len}_shard${num_shard}=${post_fix} \
  --wandb_runid 0 \
  --num_shard $num_shard \
  --num_replicate $num_replicate \
  --use_flex True \
  --ema 0.995 \
  --save_every 1000 \
  --ce_weight 0.1
