# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import json
import argparse
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

openai.api_key = os.getenv('OPENAI_API_KEY')

lock = threading.Lock()  # For thread-safe file writing

def load_prompts(prompts_json_path):
    with open(prompts_json_path, 'r') as f:
        return json.load(f)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

def call_gpt(original_image_path, result_image_path, edit_prompt, edit_type, prompts):
    try:
        original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)

        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}
        
        prompt = prompts[edit_type]
        full_prompt = prompt.replace('<edit_prompt>', edit_prompt)

        response = openai_client.chat.completions.create(
            model=model,
            stream=False,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}}
                ]
            }]
        )
        return response
    except Exception as e:
        print(f"Error in calling GPT API: {e}")
        raise

def save_result_jsonl(result, key, output_jsonl_path):
    with lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            data = {
                "key": key,
                "result": result
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_processed_keys(jsonl_path):
    processed_keys = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_keys.add(data["key"])
                except Exception as e:
                    print(f"Error loading line: {e}")
    return processed_keys

def collect_jsonl_to_dict(jsonl_path):
    result_dict = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    result_dict[data["key"]] = data["result"]
                except Exception as e:
                    print(f"Error parsing line: {e}")
    return result_dict

def process_single_item(key, item, result_img_folder, origin_img_root, prompts, output_jsonl_path):
    result_img_name = f"{key}.png"
    result_img_path = os.path.join(result_img_folder, result_img_name)
    origin_img_path = os.path.join(origin_img_root, item['id'])
    edit_prompt = item['prompt']
    edit_type = item['edit_type']

    response = call_gpt(origin_img_path, result_img_path, edit_prompt, edit_type, prompts)
    # Ensure 'choices' attribute exists in response
    result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
    save_result_jsonl(result, key, output_jsonl_path)
    return key, result

def process_json(edit_json, result_img_folder, origin_img_root, num_threads, prompts):
    output_jsonl_path = os.path.join(result_img_folder, 'result.jsonl')
    output_json_path = os.path.join(result_img_folder, 'result.json')
    with open(edit_json, 'r') as f:
        edit_infos = json.load(f)
    # Load already processed keys
    processed_keys = load_processed_keys(output_jsonl_path)
    print(f"{len(processed_keys)} items already processed, {len(edit_infos) - len(processed_keys)} remaining...")
    # Filter out tasks that have already been processed
    left_edit_infos = {k: v for k, v in edit_infos.items() if k not in processed_keys}
    total = len(left_edit_infos)
    if total == 0:
        print("Nothing to process. All items are completed.")
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_key = {
                executor.submit(process_single_item, key, item, result_img_folder, origin_img_root, prompts, output_jsonl_path): key
                for key, item in left_edit_infos.items()
            }
            for future in tqdm(as_completed(future_to_key), total=total, desc="Processing edits"):
                key = future_to_key[future]
                try:
                    future.result()  # Already saved in jsonl
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    # Failed keys will not be saved to jsonl
    # After all finished, collect jsonl to dict and save to json
    final_results = collect_jsonl_to_dict(output_jsonl_path)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"All processing completed. Final result saved in {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder with subfolders of edited images")
    parser.add_argument('--edit_json', type=str, required=True, help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    parser.add_argument('--prompts_json', type=str, required=True, help="JSON file containing prompts") 
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)  
    process_json(args.edit_json, args.result_img_folder, args.origin_img_root, args.num_processes, prompts)

if __name__ == "__main__":
    base_url = "your_api_url"
    api_version = "2024-03-01-preview"
    api_key = openai.api_key
    model = "gpt-4o-2024-11-20"
    openai_client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=api_key,
    )
    main()
