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

exp_names=(
    pre00_seed_blip3o_all_robots_jul19_gpu64seq16384_pretrain_lr5e-5
    pre01_seed_blip3o_all_robots_jul19_gpu64_seq16384_shard8_pretrain
    pre02_seed_blip3o_all_robots_jul19_t1.0_gpu64_seq16384_shard8_pretrain_lr1e-5__PRE01_25k
    pre03_seed_blip3o_all_robots_jul19_pickle_t1.0_gpu32_seq32768_shard8_pretrain_lr8e-6__PRE02_16k
    pre04_seed_blip3o_all_robots_w_notext_pickle_t1.0_gpu32_seq32768_shard8_pretrain
    pre05_seed_blip3o_all_robots_jul19_t4.0_gpu64_seq16384_shard8_pretrain
    pre06_seed_blip3o_all_50sampling_pickle_jul29_t1.0_gpu32_seq32768_shard8__PRE04_17k
    pre07_seed_blip3o_all_50sampling_pickle_jul29_t1.0_gpu16_seq32768_shard8__PRE04_17k
)       
ckpts=(0040000 0030000 0025000 0020000 0015000 0010000)

exp_names=(
    8_seed_all_robots_jul19_t1.0_gpu16_seq16384_shard8
    4_seedp1_0.2_static_mobile_allview_endspan_nolast50_t1.0_gpu16_seq16384_shard8__PRE01_8.5k

)       

# exp_names=(
#     1_pi_arxs_ur5_allview_seq_seedp1_gpu16_seq16384
#     2_pi_h1g1_allview_seq_seedp1_gpu16_seq16384
#     3_pi_arxs_ur5b0b1_allview_endspan_nolast50_seedp1_gpu16_seq16384__RUN1_70k
#     7_h1g1_vit_t1.0_gpu16_seq16384_shard8
# )       
ckpts=(0100000 0080000 0060000 0040000 0020000 0010000)


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
condition_on_vit=false


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
