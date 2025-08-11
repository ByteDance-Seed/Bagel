# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
# DATA_DIR = "/home/liliyu/workspace/monopi/monopi/experimental/liliyu/export_wm/"

GPUS=8
image_key=all_views
resolution=224
with_condition=False

mode=raw
wandb_project_name=bagel-edit-eval

# Set these to true or false to control the flags
use_vit_as_condition=false
add_vit_as_condition=false
image_list_str="image_0,image_2,image_3"

# exp_names=(
#     7_h1g1_vit_t1.0_gpu16_seq16384_shard8
# )       
# ckpts=(0070000 0060000 0040000 0020000 0010000)
# task_names=(
#     # arx_biarm_bussing_rollout
#     # arx_biarm_organize_desk_rollout
#     # arx_biarm_organize_desk_disorganize_rollout
#     # Add messy kitche ur5_biarm ??
#     h1g1_dishes_in_sink_rollout
#     h1g1_make_the_bed_rollout
#     g1h1_drawer_rollout
#     # g1h1_drawer_ood_rollout
# )
# use_vit_as_condition=true
z
task_names=(
    arx_biarm_bussing_rollout
    arx_biarm_organize_desk_rollout
    arx_biarm_organize_desk_disorganize_rollout
    # Add messy kitche ur5_biarm ??
    h1g1_dishes_in_sink_rollout
    h1g1_make_the_bed_rollout
    g1h1_drawer_rollout
    # g1h1_drawer_ood_rollout
)
ckpts=(0020000 0010000) # 0015000)
# ckpts=(0015000)#
exp_names=(
    # 9_seed_blip3o_all_75sampling_pickle_t1.0_gpu16_seq16384_shard8
    10_seed_blip3o_all_75sampling_with_vit_pickle_t1.0_gpu16_seq16384_shard8
    11_seed_blip3o_all_75sampling_with_vit_textloss_pickle_t1.0_gpu16_seq16384_shard8
)       
add_vit_as_condition=true

for ckpt in "${ckpts[@]}"; do
    for task_name in "${task_names[@]}"; do
        for exp_name in "${exp_names[@]}"; do
            # Build command with optional flags
            cmd="PYTHONPATH=. torchrun \
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
                --wandb_project_name $wandb_project_name"
            
            # Add flags conditionally
            if [ "$use_vit_as_condition" = true ]; then
                cmd="$cmd --use_vit_as_condition"
            fi
            if [ "$add_vit_as_condition" = true ]; then
                cmd="$cmd --add_vit_as_condition"
            fi
            
            # Execute the command
            eval $cmd 
                # --with_condition $with_condition
        done
    done
done
