# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export OPENAI_API_KEY=$openai_api_key

GPUS=8


# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_imgedit.py \
    --output_dir $output_path/bagel \
    --metadata_file ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --max_latent_size 64 \
    --model-path $model_path


# calculate score
python ./eval/gen/imgedit/basic_bench.py \
    --result_img_folder $output_path/bagel \
    --edit_json ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --origin_img_root ./eval/gen/imgedit/Benchmark/singleturn \
    --num_processes 4 \
    --prompts_json ./eval/gen/imgedit/Benchmark/singleturn/judge_prompt.json


# summarize score
python ./eval/gen/imgedit/step1_get_avgscore.py \
    --result_json $output_path/bagel/result.json \
    --average_score_json $output_path/bagel/average_score.json

python ./eval/gen/imgedit/step2_typescore.py \
    --average_score_json  $output_path/bagel/average_score.json \
    --edit_json ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --typescore_json $output_path/bagel/typescore.json