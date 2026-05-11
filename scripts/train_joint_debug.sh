#!/bin/bash
#SBATCH -J train_debug
#SBATCH -p efm_t
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=bageltest
#SBATCH --quota=reserved
#SBATCH --nodes=1
#SBATCH --debug

# set -x -e
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5_0

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 25188))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
echo $MASTER_ADDR
echo $MASTER_PORT

# export WORLD_SIZE=$SLURM_NTASKS

export NNODES=1
export num_gpus=8

name="train0828_joint"
export output_dir="/mnt/inspurfs/mozi_t/huwenbo/bagel_output/train_checkpoints/${name}/"
mkdir -p ${output_dir}
export checkpoint_dir="/mnt/inspurfs/mozi_t/huwenbo/bagel_output/train_checkpoints/${name}"
mkdir -p ${checkpoint_dir}


# export PYTHONPATH="${PYTHONPATH}:/mnt/petrelfs/huwenbo/vl_res/Bagel"
export PYTHONPATH="${PYTHONPATH}:/mnt/petrelfs/huwenbo/vl_res/mllm3r"

export HF_HOME="/mnt/petrelfs/huwenbo/.cache/huggingface"

# export WANDB_SSL_VERIFY=false
# export WANDB_MODE=online  
export WANDB_MODE=offline 
export WANDB_API_KEY=ae4d28f3fe1425db4be97835fc49e27c3029731f
export WANDB_KEY=ae4d28f3fe1425db4be97835fc49e27c3029731f

export current_time=$(date +%Y%m%d_%H%M%S)
export wandb_name="run_bagel1.5B_qwenvl2_debug"

srun -p mozi_t  apptainer exec --nv  --bind /mnt:/mnt/ /mnt/petrelfs/huwenbo/vl_res/ubuntu20.04-py3.10-cuda11.8-cudnn8-wenbo_t.sif \
    bash -c 'torchrun \
    --nnodes $NNODES \
    --nproc_per_node ${num_gpus:-1} \
    --node_rank="${SLURM_NODEID}" \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train/pretrain_unified_navit.py \
    --dataset_config_file /mnt/petrelfs/huwenbo/vl_res/mllm3r/data/configs/joint_train_debug.yaml \
    --layer_module Qwen2VLMoTDecoderLayer \
    --vit_path /mnt/inspurfs/mozi_t/huwenbo/weights/qwenvl_2b_bagel_new \
    --dino_path facebook/dinov2-with-registers-large \
    --llm_path /mnt/inspurfs/mozi_t/huwenbo/weights/qwenvl_2b_bagel_new \
    --model_path /mnt/inspurfs/mozi_t/huwenbo/weights/qwenvl_2b_bagel_new \
    --use_flex True \
    --expected_num_tokens 4096 \
    --max_num_tokens 4096 \
    --max_num_tokens_per_sample 4096 \
    --wandb_project bagel_recon \
    --wandb_name ${wandb_name} \
    --wandb_offline True \
    --wandb_resume allow \
    --checkpoint_dir ${checkpoint_dir} \
    --llm_qk_norm True \
    --finetune_from_hf True \
    --auto_resume False \
    --resume-model-only True \
    --finetune-from-ema True \
    --resume_from /mnt/inspurfs/mozi_t/huwenbo/weights/qwenvl_2b_bagel_new \
    --finetune_dino_from_hf False \
    --copy_init_moe False \
    --visual_gen False \
    --visual_und True \
    --visual_recon True \
    --freeze_dino True \
    --freeze_vit True \
    --freeze_und True \
    --results_dir $output_dir \
    --max_latent_size 64  \
    --save_every 10000 \
    --total_steps 20000 \
    --warmup_steps 1000 \
    --log_every 1 \
    --num_shard 8 \
    --lr 1e-4 \
    --lr_scheduler cosine \
    --num_workers 1'
    
