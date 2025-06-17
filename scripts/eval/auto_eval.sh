# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-small-fake
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT

root_dir=/home/liliyu/workspace/BAGEL/results

GPUS=8
exp_names=(
    # pi_arx_100step_seedp1_gpu8_seq32768
    # pi_arx_50step_seedp1_448_gpu16_seq32768
    # pi_arx_endspan_seedp1_448_gpu8_seq16384
    # pi_arx_50step_seedp1_gpu8_seq16384
    pi_arx_i2_endspan_seedp1_gpu8_seq16384
)
task_name='arx_endspan'
image_key="image_2"

# exp_names=(
#     pi_ur5_endspan_seedp1_448_gpu8_seq32768
# )
# task_name='ur5_endspan'


# while true; do
for mode in raw ema; do
    for exp_name in "${exp_names[@]}"; do
        if [[ $exp_name == *"448"* ]]; then
            resolution=448
        else
            resolution=224
        fi
        echo ${exp_name}
        exp_dir=${root_dir}/${exp_name}
        ckpt_dir=${exp_dir}/checkpoints/*
        for d in ${ckpt_dir}; do
            echo ${d}
            ckpt=$(basename ${d})
            # # Edit images
            step=${ckpt##*_}
            step=$((10#$step))
            echo ${step}
            
            if (( $step % 10000 != 0 || $step <= 0 ))
            then
                continue
            fi
            echo $exp_name $mode $resolution $ckpt
            PYTHONPATH=. torchrun \
                --nnodes=1 \
                --node_rank=0 \
                --nproc_per_node=$GPUS \
                --master_addr=127.0.0.1 \
                --master_port=12345 \
                ./eval/gen/gen_images_edit_mp_new.py \
                --model-path $model_path \
                --task_name $task_name \
                --image_key $image_key \
                --resolution $resolution \
                --run_name $exp_name  \
                --checkpoint_step ${ckpt} \
                --model_mode $mode 
        done

    done
    
done
#     echo "rest"
#     sleep 1h
# done

# # Edit images
# PYTHONPATH=. torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr=127.0.0.1 \
#     --master_port=12345 \
#     ./eval/gen/gen_images_edit_mp.py \
#     --model-path $model_path \
#     --resolution 448 \
#     --run_name pi_arx_50step_seed_448_gpu8_seq16384 \
#     --checkpoint_step 0015000 \
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