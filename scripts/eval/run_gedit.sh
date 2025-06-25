# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# run this script at the root of the project folder
pip install httpx==0.23.0
pip install openai==1.87.0
pip install datasets
pip install megfile


N_GPU=8  # Number of GPU used in for the evaluation
MODEL_PATH="/Path/to/BAGEL-7B-MoT"
OUTPUT_DIR="/Path/to/save/results"
GEN_DIR="$OUTPUT_DIR/gen_image"
LOG_DIR="$OUTPUT_DIR/logs"

AZURE_ENDPOINT="https://azure_endpoint_url_you_use"  # set up the azure openai endpoint url
AZURE_OPENAI_KEY=""  # set up the azure openai key
N_GPT_PARALLEL=10


mkdir -p "$OUTPUT_DIR"
mkdir -p "$GEN_DIR"
mkdir -p "$LOG_DIR"


# # ----------------------------
# #    Download GEdit Dataset
# # ----------------------------
python -c "from datasets import load_dataset; dataset = load_dataset('stepfun-ai/GEdit-Bench')"
echo "Dataset Downloaded"


# # ---------------------
# #    Generate Images
# # ---------------------
for ((i=0; i<$N_GPU; i++)); do
    nohup python3 eval/gen/gedit/gen_images_gedit.py --model_path "$MODEL_PATH"  --output_dir "$GEN_DIR"  --shard_id $i --total_shards "$N_GPU" --device $i  2>&1 | tee "$LOG_DIR"/request_$(($N_GPU + i)).log &
done

wait
echo "Image Generation Done"


# # ---------------------
# #    GPT Evaluation
# # ---------------------
cd eval/gen/gedit
python test_gedit_score.py --save_path "$OUTPUT_DIR" --azure_endpoint "$AZURE_ENDPOINT" --gpt_keys "$AZURE_OPENAI_KEY"  --max_workers "$N_GPT_PARALLEL"
echo "Evaluation Done"


# # --------------------
# #    Print Results
# # --------------------
python calculate_statistics.py --save_path "$OUTPUT_DIR"  --language en

