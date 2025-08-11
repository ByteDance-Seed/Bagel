# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
# DATA_DIR = "/mnt/weka/liliyu/export_wm/"

GPUS=8
image_key=all_views
with_condition=False

mode=raw
wandb_project_name=bagel-edit-eval

exp_names=(
    pre08_seed_blip3o_robot_256_320px_t1.0_gpu16_seq32768_shard8__PRE02_16k
)       
exp_names=(
    pre09_seed_blip3o_robot_448px_t1.0_gpu16_seq32768_shard8__PRE02_16k
)   
resolution=448
ckpts=(0030000 0025000 0020000 0015000 0010000)



image_list_str="image_0,image_2,image_3"
task_names=(
    arx_biarm_bussing_rollout
    arx_biarm_organize_desk_rollout
    # arx_biarm_organize_desk_disorganize_rollout
    # Add messy kitche ur5_biarm ??
    h1g1_dishes_in_sink_rollout
    h1g1_make_the_bed_rollout
    g1h1_drawer_rollout
    # g1h1_drawer_ood_rollout
)
use_vit_as_condition=false


for ckpt in "${ckpts[@]}"; do
    for task_name in "${task_names[@]}"; do
        for exp_name in "${exp_names[@]}"; do
            PYTHONPATH=. torchrun \
                --nnodes=1 \
                --node_rank=0 \
                --nproc_per_node=$GPUS \
                --master_addr=127.0.0.1 \
                --master_port=12345 \
                ./eval/gen/gen_images_edit_allviews_ddp.py \
                --model-path $model_path \
                --task_name $task_name \
                --image_list_str $image_list_str \
                --resolution $resolution \
                --run_name $exp_name  \
                --checkpoint_step ${ckpt} \
                --model_mode $mode  \
                --wandb_project_name $wandb_project_name 
                # --with_condition $with_condition
        done
    done
done
