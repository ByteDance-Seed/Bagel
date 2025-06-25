# Copyright (c) 2025 mercurystraw
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-06-15.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/mercurystraw/Kris_Bench/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import json
import base64
import time
import logging
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from prompts import (
    prompt_consist,
    prompt_quality,
    prompt_view_instruction_following,
    prompt_instruction_following
)
from metrics_common import (
    extract_consistency_score_and_reason,
    extract_instruction_score_and_reason,
    extract_quality_score_and_reason
)

lock = threading.Lock()  # Thread-safe file writing lock
openai.api_key = os.getenv('OPENAI_API_KEY')

def save_result_jsonl(result, key, output_jsonl_path):
    """Save evaluation result to JSONL file with thread lock"""
    with lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            data = {"key": key, "result": result}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_processed_keys_with_missing_metrics(jsonl_path, metrics, expected_keys_map):
    """Load processed image IDs and return missing metrics for each key"""
    key_missing_metrics = {}  # key -> list of missing metrics
    fully_completed_keys = set()  # keys that have all metrics completed
    
    if os.path.exists(jsonl_path):
        # First, collect all results for each key
        key_results = {}  # key -> merged result dict
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = data["key"]
                    result = data["result"]
                    
                    if key not in key_results:
                        key_results[key] = {}
                    
                    # Merge results (later entries can overwrite earlier ones)
                    key_results[key].update(result)
                    
                except Exception as e:
                    print(f"Error loading line: {e}")
        
        # Now check which metrics are missing for each key
        for key, merged_result in key_results.items():
            missing_metrics = []
            
            for metric in metrics:
                if metric in expected_keys_map:
                    metric_complete = True
                    for score_key in expected_keys_map[metric]:
                        if merged_result.get(score_key) is None:
                            metric_complete = False
                            break
                    
                    if not metric_complete:
                        missing_metrics.append(metric)
            
            if missing_metrics:
                key_missing_metrics[key] = missing_metrics
            else:
                fully_completed_keys.add(key)
    
    return key_missing_metrics, fully_completed_keys

def collect_jsonl_to_dict(jsonl_path, metrics, expected_keys_map):
    """Convert JSONL file to dictionary, merging same keys and filtering incomplete results"""
    result_dict = {}
    
    if os.path.exists(jsonl_path):
        # First, collect and merge all results for each key
        key_results = {}  # key -> merged result dict
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = data["key"]
                    result = data["result"]
                    
                    if key not in key_results:
                        key_results[key] = {}
                    
                    # Merge results (later entries can overwrite earlier ones)
                    key_results[key].update(result)
                    
                except Exception as e:
                    print(f"Error parsing line: {e}")
        
        # Now filter based on completeness
        for key, merged_result in key_results.items():
            all_metrics_complete = True
            incomplete_metrics = []
            
            for metric in metrics:
                if metric in expected_keys_map:
                    for score_key in expected_keys_map[metric]:
                        if merged_result.get(score_key) is None:
                            all_metrics_complete = False
                            incomplete_metrics.append(f"{metric}({score_key})")
            
            if all_metrics_complete:
                result_dict[key] = merged_result
            else:
                # Log incomplete results for debugging
                logging.info(f"Incomplete result for {key}: missing {', '.join(incomplete_metrics)}")
    
    return result_dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True)
parser.add_argument("--models", type=str, nargs="+", default=["bagel"])
parser.add_argument("--max_workers", type=int, default=8)
args = parser.parse_args()

# Constants
BENCH_DIR = "eval/gen/kris/KRIS_Bench"
RESULTS_DIR = args.results_dir
MODELS = args.models
CATEGORIES = ["viewpoint_change"]
METRICS = ["consistency", "instruction_following", "image_quality"]

# Initialize OpenAI client
api_key = openai.api_key
base_url = "your_api_url"
api_version = "2024-03-01-preview"
openai_client = openai.AzureOpenAI(
    azure_endpoint=base_url,
    api_version=api_version,
    api_key=api_key,
)

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logging.error("Error encoding image %s: %s", image_path, e)
        return None

def evaluate_with_gpt(prompt, original_b64=None, edited_b64=None, gt_b64=None):
    """Call GPT with images/text and retry on failure."""
    message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    
    if original_b64:
        message["content"].extend([
            {"type": "text", "text": "This is the original image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}}
        ])
    if edited_b64:
        message["content"].extend([
            {"type": "text", "text": "This is the edited image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_b64}"}}
        ])
    if gt_b64:
        message["content"].extend([
            {"type": "text", "text": "This is the ground truth image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"}}
        ])

    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[message],
                stream=False,
                max_tokens=1000
            )
            return resp.choices[0].message.content
        except Exception as e:
            logging.warning(f"GPT call failed (attempt {attempt+1}/3): {e}")
            time.sleep(5)
    logging.error("GPT evaluation failed after 3 attempts.")
    return ""

def evaluate_images(model_name, category, image_id, metrics=None):
    """
    Evaluate images for a specific model, category and image ID.
    Returns a dict with score and reasoning for each metric.
    """
    metrics = metrics or METRICS
    results = {}

    annotation_path = os.path.join(BENCH_DIR, category, "annotation.json")
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except Exception as e:
        logging.error("Failed to load annotation file %s: %s", annotation_path, e)
        return results

    ann = annotations.get(str(image_id))
    if not ann:
        logging.warning("Image ID %s not found in %s", image_id, annotation_path)
        return results

    # Build paths
    ori_path = os.path.join(BENCH_DIR, category, ann["ori_img"])
    edited_path = os.path.join(RESULTS_DIR, model_name, category, f"{image_id}.png")
    gt_rel = ann.get("gt_img")
    gt_path = os.path.join(BENCH_DIR, category, gt_rel) if gt_rel else None

    # Validate files
    if not os.path.exists(ori_path):
        logging.error("Original image missing: %s", ori_path)
        return results
    if not os.path.exists(edited_path):
        logging.error("Edited image missing: %s", edited_path)
        return results
    if category == "viewpoint_change" and (not gt_path or not os.path.exists(gt_path)):
        logging.error("Ground-truth image missing: %s", gt_path)
        return results

    # Encode images
    ori_b64 = encode_image_to_base64(ori_path)
    edt_b64 = encode_image_to_base64(edited_path)
    gt_b64 = encode_image_to_base64(gt_path) if gt_path else None
    if not ori_b64 or not edt_b64 or (category == "viewpoint_change" and not gt_b64):
        logging.error("Failed to encode images for %s/%s/%s", model_name, category, image_id)
        return results

    instr = ann.get("ins_en", "")
    for metric in metrics:
        if metric == "consistency":
            prompt = prompt_consist.format(instruct=instr)
            resp = evaluate_with_gpt(prompt, ori_b64, edt_b64)
            score, reason = extract_consistency_score_and_reason(resp)
            results["consistency_score"] = score
            results["consistency_reasoning"] = reason
        elif metric == "instruction_following":
            if category == "viewpoint_change":
                prompt = prompt_view_instruction_following.format(instruct=instr)
                resp = evaluate_with_gpt(prompt, ori_b64, edt_b64, gt_b64)
            else:
                prompt = prompt_instruction_following.format(instruct=instr)
                resp = evaluate_with_gpt(prompt, ori_b64, edt_b64)
            score, reason = extract_instruction_score_and_reason(resp)
            results["instruction_score"] = score
            results["instruction_reasoning"] = reason
        elif metric == "image_quality":
            resp = evaluate_with_gpt(prompt_quality, None, edt_b64)
            score, reason = extract_quality_score_and_reason(resp)
            results["quality_score"] = score
            results["quality_reasoning"] = reason
        else:
            logging.warning("Unknown metric: %s", metric)

    return results

def process_image_eval(model, category, image_id, metrics, annotations, output_jsonl_path):
    """Helper for threaded evaluation of a single image."""
    results = evaluate_images(model, category, image_id, metrics)
    if not results:
        return None
    
    ann = annotations.get(image_id, {})
    data = {
        "instruction": ann.get("ins_en", ""),
        "explain": ann.get("explain_en", ""),
        **results
    }
    save_result_jsonl(data, image_id, output_jsonl_path)
    return image_id, data

def run_evaluation(models=None, categories=None, metrics=None):
    """Main evaluation process with multi-threading"""
    models = models or MODELS
    categories = categories or CATEGORIES
    metrics = metrics or METRICS
    
    # mapping of metric to expected result keys
    expected_keys_map = {
        "consistency":          ["consistency_score"],
        "instruction_following": ["instruction_score"],
        "image_quality":        ["quality_score"],
    }

    for model in models:
        for category in tqdm(categories, desc=f"Evaluating {model}"):
            ann_file = os.path.join(BENCH_DIR, category, "annotation.json")
            if not os.path.isfile(ann_file):
                logging.error(f"Missing annotation.json: {ann_file}")
                continue

            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            except Exception as e:
                logging.error(f"Error reading annotations {ann_file}: {e}")
                continue

            image_ids = list(annotations.keys())
            out_dir = os.path.join(RESULTS_DIR, model, category)
            os.makedirs(out_dir, exist_ok=True)
            metrics_file = os.path.join(out_dir, "metrics.json")
            metrics_jsonl = os.path.join(out_dir, "metrics.jsonl")
            
            # Get missing metrics for each key and fully completed keys
            key_missing_metrics, fully_completed_keys = load_processed_keys_with_missing_metrics(
                metrics_jsonl, metrics, expected_keys_map
            )
            
            # Build list of images that need processing
            to_process = []
            for img_id in image_ids:
                if img_id in key_missing_metrics:
                    # This image has some missing metrics
                    missing_metrics = key_missing_metrics[img_id]
                    to_process.append((img_id, missing_metrics))
                elif img_id not in fully_completed_keys:
                    # This image hasn't been processed at all
                    to_process.append((img_id, metrics))
                # If img_id in fully_completed_keys, skip it (already fully completed)
            
            if not to_process:
                logging.info(f"No images to process for {model}/{category}. All {len(fully_completed_keys)} images are fully completed.")
            else:
                logging.info(f"Processing {len(to_process)} images for {model}/{category}. {len(fully_completed_keys)} images already completed.")
                
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    futures = {
                        executor.submit(process_image_eval, model, category, img_id, img_metrics, annotations, metrics_jsonl): (img_id, img_metrics)
                        for img_id, img_metrics in to_process
                    }
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{model}/{category}", leave=False):
                        try:
                            result = fut.result()
                            if result:
                                img_id, img_metrics = futures[fut]
                                logging.debug(f"Completed {img_id} with metrics {img_metrics}")
                        except Exception as e:
                            img_id, img_metrics = futures[fut]
                            logging.error(f"Failed processing {img_id} with metrics {img_metrics}: {e}")
                            
            try:
                # Collect final results (only complete ones)
                metrics_data = collect_jsonl_to_dict(metrics_jsonl, metrics, expected_keys_map)
                with open(metrics_file, "w", encoding="utf-8") as wf:
                    json.dump(metrics_data, wf, ensure_ascii=False, indent=2)
                logging.info(f"Saved {len(metrics_data)} complete results to {metrics_file}")
            except Exception as e:
                logging.error(f"Failed to save metrics to {metrics_file}: {e}")

if __name__ == "__main__":
    run_evaluation()
