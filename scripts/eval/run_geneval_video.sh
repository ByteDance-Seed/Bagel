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

# exp_name=pi_arx_single_allview_seq_seedp1_gpu16_seq16384
# task_name=arx_endspan_lang
# image_list_str="image_0,image_2"

# exp_name=pi_ur5e4_b0_allview_seq_seedp1_gpu16_seq16384
# task_name=ur5e4_endspan_lang
# image_list_str="image_0,image_2"

# exp_name=pi_ur5e4_b1_allview_seq_seedp1_gpu16_seq16384
# task_name=ur5e4_b1_endspan_lang
# image_list_str="image_0,image_2"
ckpt=0019000

# exp_name=pi_diverse_batch_100steps_seedp1_gpu16_seq16384
# task_name=diverse_batch_folding_step100
# image_list_str="image_0,image_2,image_3"
# ckpt=0047500

# exp_name=pi_diverse_batch_conditioning_100steps_seedp1_gpu16_seq16384
# task_name=diverse_batch_folding_step100
# image_list_str="image_0,image_2,image_3"
# ckpt=0050000
# with_condition=True

exp_name=pi_arx_biarm_allview_seq_seedp1_gpu16_seq16384
task_name=diverse_batch_folding_step100
image_list_str="image_0,image_2,image_3"
ckpt=0050000


# exp_name=pi_arxs_ur5_allview_seq_seedp1_gpu16_seq16384   
# # task_name=arx_endspan_lang
# # image_list_str="image_0,image_2"
# # task_name=ur5e4_endspan_lang_1
# task_name=ur5_bus_generalization_rollout
# image_list_str="image_0,image_2"
# ckpt=0070000

# exp_name=pi_ur5e4_b1_allview_seq_seedp1_gpu16_seq16384   
# exp_name=pi_arxs_ur5_allview_seq_seedp1_gpu16_seq16384  
exp_name=pi_arxs_ur5_allview_endspan_nolast50_seedp1_gpu16_seq16384
exp_name=seed_blip3o_all_robots_jul19_gpu64seq16384_pretrain
exp_name=seed_blip3o_all_robots_jul18_gpu64seq16384_pretrain
exp_name=seedp1_0.2_static_mobile_allview_endspan_nolast50_t1.0_gpu16_seq16384_shard8__
exp_name=seed_blip3o_all_robots_jul19_gpu64_seq16384_shard8_pretraintest
exp_name=seed_blip3o_all_robots_jul19_t4.0_gpu64_seq16384_shard8_pretrain
# task_name=arx_biarm_endspan_lang
image_list_str="image_0,image_2,image_3"
task_names=(
    arx_biarm_bussing_rollout
    arx_biarm_organize_desk_rollout
    # arx_biarm_organize_desk_disorganize_rollout
    # Add messy kitche ur5_biarm ??
    h1g1_dishes_in_sink_rollout
    # h1g1_make_the_bed_rollout
    # g1h1_drawer_rollout
    # g1h1_drawer_ood_rollout
)
ckpt=0005000
ckpt=0014000
ckpt=0040000
ckpt=0014000

# # exp_name=pi_h1g1_allview_seq_seedp1_gpu16_seq16384   
# task_names=(
#     h1g1_dishes_in_sink_rollout
#     h1g1_make_the_bed_rollout
#     g1h1_drawer_rollout
#     g1h1_drawer_ood_rollout
# )
# image_list_str="image_0,image_2,image_3"
# ckpt=0070000

for task_name in "${task_names[@]}"; do
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

# GPUS=4
# step=0002000
# # Edit images
# PYTHONPATH=. torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr=127.0.0.1 \
#     --master_port=12345 \
#     ./eval/gen/gen_images_edit_ddp.py \
#     --model-path $model_path \
#     --resolution 448 \
#     --checkpoint_step $step \
#     --run_name pi_ur5_endspan_seedp1_gpu8_seq32768 \
#     --model_mode raw


# # Edit images
# PYTHONPATH=. torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr=127.0.0.1 \
#     --master_port=12345 \
#     simple_edits.py \
#     --model-path $model_path \
#     # --resolution 448 \
#     # --run_name pi_arx_50step_seed_448_gpu8_seq16384 \
#     # --checkpoint_step 0002000 \
#     # --model_mode ema




# # generate images
# PYTHONPATH=. torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr=127.0.0.1 \\

#     --master_port=12345 \
#     ./eval/gen/gen_images_mp.py \
#     --output_dir generated/images \/
#     --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \
#     --batch_size 2 \
#     --num_images 2 \
#     --resolution 256 \
#     --max_latent_size 64 \
#     --model-path $model_path \
#     # --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \


# # calculate score
# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \s
#     --master_addr=127.0.0.1 \
#     --master_port=12345 \ 
#     ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
#     $OUTPUT_DIR/images \
#     --outfile $OUTPUT_DIR/results.jsonl \
#     --model-path ./eval/gen/geneval/model


# # summarize score
# python ./eval/gen/geneval/evaluation/summary_scores.py $OUTPUT_DIR/results.jsonl
