# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT

GPUS=8
task_name=teleop_direct_448
# image_key=image_0
image_keys=(image_1)
resolution=448
exp_name=pretrained
ckpt=-1
mode=ema
wandb_project_name=bagel-zero-shot
ids_for_fixed_prompts=(2)
cfg_text_scales=(4.0)
cfg_img_scales=(1.2)


for image_key in ${image_keys[@]}; do
    for cfg_img_scale in ${cfg_img_scales[@]}; do
        for cfg_text_scale in ${cfg_text_scales[@]}; do
            for ids_for_fixed_prompt in ${ids_for_fixed_prompts[@]}; do
                PYTHONPATH=. torchrun \
                    --nnodes=1 \
                    --node_rank=0 \
                    --nproc_per_node=$GPUS \
                    --master_addr=127.0.0.1 \
                    --master_port=12345 \
                    ./eval/gen/gen_images_edit_ddp.py \
                    --model-path $model_path \
                    --task_name $task_name \
                    --image_key $image_key \
                    --resolution $resolution \
                    --run_name $exp_name  \
                    --checkpoint_step ${ckpt} \
                    --model_mode $mode  \
                    --wandb_project_name $wandb_project_name \
                    --ids_for_fixed_prompt $ids_for_fixed_prompt \
                    --cfg_text_scale $cfg_text_scale \
                    --cfg_img_scale $cfg_img_scale
            done
        done
    done
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
