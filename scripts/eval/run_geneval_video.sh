# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT

GPUS=8
image_key=all_views
resolution=224

# exp_name=pi_arx_single_allview_seq_seedp1_gpu16_seq16384
# task_name=arx_endspan_lang
# image_list_str="image_0,image_2"

# exp_name=pi_ur5e4_b0_allview_seq_seedp1_gpu16_seq16384
# task_name=ur5e4_endspan_lang
# image_list_str="image_0,image_2"

# exp_name=pi_ur5e4_b1_allview_seq_seedp1_gpu16_seq16384
# task_name=ur5e4_b1_endspan_lang
# image_list_str="image_0,image_2"
ckpt=0020000

exp_name=pi_arx_biarm_allview_seq_seedp1_gpu16_seq16384
task_name=arx_biarm_endspan_lang
image_list_str="image_0,image_2,image_3"
ckpt=0015000

mode=raw
wandb_project_name=bagel-edit-eval


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
