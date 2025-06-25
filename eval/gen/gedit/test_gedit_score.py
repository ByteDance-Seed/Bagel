# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from viescore import VIEScore
import PIL
import os
import megfile
from PIL import Image
from tqdm import tqdm
import json
from datasets import load_dataset
import sys
import csv
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
GROUPS = [
    "background_change", "color_alter", "material_alter", "motion_change", "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"
]

model_name = 'bagel'

def process_single_item(item, vie_score, max_retries=10000):

    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    intersection_exist = item['Intersection_exist']
    sample_prefix = key
    save_path_fullset_source_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}_SRCIMG.png"
    save_path_fullset_result_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}.png"
    
    src_image_path = save_path_fullset_source_image
    save_path_item = save_path_fullset_result_image
    
    for retry in range(max_retries):
        try:
            pil_image_raw =Image.open(megfile.smart_open(src_image_path, 'rb'))
            pil_image_edited = Image.open(megfile.smart_open(save_path_item, 'rb')).convert("RGB").resize((pil_image_raw.size[0], pil_image_raw.size[1]))

            text_prompt = instruction
            score_list = vie_score.evaluate([pil_image_raw, pil_image_edited], text_prompt)
            sementics_score, quality_score, overall_score = score_list

            # print(f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}")
            
            return {
                "source_image": src_image_path,
                "edited_image": save_path_item,
                "instruction": instruction,
                "sementics_score": sementics_score,
                "quality_score": quality_score,
                "intersection_exist" : item['Intersection_exist'],
                "instruction_language" : item['instruction_language']
            }
        except Exception as e:
            
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # 指数退避：2秒, 4秒, 6秒...
                print(f"Error processing {save_path_item} (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process {save_path_item} after {max_retries} attempts: {e}")
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="gpt4o", choices=["gpt4o", "qwen25vl"])
    parser.add_argument("--gpt_keys", type=str, required=True, nargs='+')
    parser.add_argument("--azure_endpoint", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=30)
    args = parser.parse_args()
    save_path_dir = args.save_path
    evaluate_group = [model_name]
    backbone = args.backbone

    vie_scores = [VIEScore(backbone=backbone, task="tie", key_path=k, azure_endpoint=args.azure_endpoint) for k in args.gpt_keys]
    dataset = load_dataset("stepfun-ai/GEdit-Bench")['train'].remove_columns(['input_image_raw', 'input_image'])

    for model_name in evaluate_group:
        save_path = os.path.join(save_path_dir, 'gen_image')
        save_path_new = os.path.join(save_path_dir, backbone, "eval_results_new")
        all_csv_list = []  # Store all results for final combined CSV
        
        # Load existing processed samples from final CSV if it exists
        processed_samples = set()
        final_csv_path = os.path.join(save_path_new, f"{model_name}_combined_gpt_score.csv")
        if megfile.smart_exists(final_csv_path):
            with megfile.smart_open(final_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create a unique identifier for each sample
                    sample_key = (row['source_image'], row['edited_image'])
                    processed_samples.add(sample_key)
            print(f"Loaded {len(processed_samples)} processed samples from existing CSV")

        for group_name in GROUPS:
            group_csv_list = []
            group_dataset_list = []  
            for item in tqdm(dataset, desc=f"Processing {model_name} - {group_name}"):
                if item['task_type'] == group_name:
                    group_dataset_list.append(item)
            # Load existing group CSV if it exists
            group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_gpt_score.csv")
            if megfile.smart_exists(group_csv_path):
                with megfile.smart_open(group_csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    group_results = list(reader)
                    group_csv_list.extend(group_results)
            
                print(f"Loaded existing results for {model_name} - {group_name}")
            
            print(f"Processing group: {group_name}")
            print(f"Processing model: {model_name}")
            
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                for i, item in enumerate(group_dataset_list):
                    instruction = item['instruction']
                    key = item['key']
                    instruction_language = item['instruction_language']
                    intersection_exist = item['Intersection_exist']
                    sample_prefix = key
                    save_path_fullset_source_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}_SRCIMG.png"
                    save_path_fullset_result_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}.png"

                    if not megfile.smart_exists(save_path_fullset_result_image) or not megfile.smart_exists(save_path_fullset_source_image):
                        print(f"Skipping {sample_prefix}: Source or edited image does not exist")
                        continue

                    # Check if this sample has already been processed
                    sample_key = (save_path_fullset_source_image, save_path_fullset_result_image)
                    exists = sample_key in processed_samples
                    if exists:
                        print(f"Skipping already processed sample: {sample_prefix}")
                        continue

                    future = executor.submit(process_single_item, item, vie_scores[i%len(vie_scores)])
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model_name} - {group_name}"):
                    result = future.result()
                    if result:
                        group_csv_list.append(result)

            # Save group-specific CSV
            group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_gpt_score.csv")
            with megfile.smart_open(group_csv_path, 'w', newline='') as f:
                fieldnames = ["source_image", "edited_image", "instruction", "sementics_score", "quality_score", "intersection_exist", "instruction_language"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in group_csv_list:
                    writer.writerow(row)
            all_csv_list.extend(group_csv_list)

            print(f"Saved group CSV for {group_name}, length: {len(group_csv_list)}")

        # After processing all groups, calculate and save combined results
        if not all_csv_list:
            print(f"Warning: No results for model {model_name}, skipping combined CSV generation")
            continue

        # Save combined CSV
        combined_csv_path = os.path.join(save_path_new, f"{model_name}_combined_gpt_score.csv")
        with megfile.smart_open(combined_csv_path, 'w', newline='') as f:
            fieldnames = ["source_image", "edited_image", "instruction", "sementics_score", "quality_score", "intersection_exist", "instruction_language"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_csv_list:
                writer.writerow(row)

                
            
            
