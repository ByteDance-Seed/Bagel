# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from collections import defaultdict

# Benchmark type and metric definitions
kris_benchamrk_type = {
    "Attribute Perception": [
        "count_change", "color_change", "size_adjustment", "part_completion", "anomaly_correction",
    ],
    "Spatial Perception": [
        "viewpoint_change", "position_movement", 
    ],
    "Temporal Prediction": [
        "temporal_prediction"
    ],
    "Social Science": [
        "humanities", "practical_knowledge"
    ],
    "Natural Science": [
        "biology", "chemistry", "geography", "medicine", "mathematics",  "physics"
    ],
    "Logical Reasoning": [
        "abstract_reasoning", "rule-based_reasoning",
    ],
    "Instruction Decomposition": [
        "multi-element_composition", "multi-instruction_execution",
    ]
}

kris_benchamrk_metric = {
    "Attribute Perception": [
        "consistency_score", "quality_score", "instruction_score",
    ],
    "Spatial Perception": [
        "consistency_score", "quality_score", "instruction_score",
    ],
    "Temporal Prediction": [
        "consistency_score", "quality_score", "instruction_score",
    ],
    "Social Science": [
        "consistency_score", "quality_score", "instruction_score", "knowledge_score",
    ],
    "Natural Science": [
        "consistency_score", "quality_score", "instruction_score", "knowledge_score",
    ],
    "Logical Reasoning": [
        "consistency_score", "quality_score", "instruction_score", "knowledge_score",
    ],
    "Instruction Decomposition": [
        "consistency_score", "quality_score", "instruction_score",
    ]
}

# Score short names and their order
score_name = {
    "consistency_score": "VC",
    "quality_score": "VQ",
    "instruction_score": "IF",
    "knowledge_score": "KP",
    "average_score": "AVG",
}
score_name_order = ["consistency_score", "quality_score", "instruction_score", "knowledge_score", "average_score"]

# Meta categories and their order
meta_categories = {
    "Factual Knowledge": [
        "Attribute Perception", "Spatial Perception", "Temporal Prediction"
    ],
    "Conceptual Knowledge": [
        "Social Science", "Natural Science"
    ],
    "Procedural Knowledge": [
        "Logical Reasoning", "Instruction Decomposition"
    ]
}
meta_category_order = list(meta_categories.keys())

def normalize_score(score):
    """Normalize a score to a 100-point scale."""
    return (score - 1) * 25 if score is not None else None

def summarize_benchmark_scores_with_normalization(results_dir):
    """
    Calculate normalized benchmark scores for each category and subitem.
    Only non-None scores are included in results.
    """
    type_all_scores = defaultdict(lambda: defaultdict(list))  # {category: {score_type: [scores]}}
    type_all_values_flat = defaultdict(list)  # For category overall AVG
    subitem_summary = defaultdict(dict)  # {category: {subitem: {score_type: avg}}}

    for b_type, sub_items in kris_benchamrk_type.items():
        metrics = kris_benchamrk_metric[b_type]
        for sub_item in sub_items:
            folder = os.path.join(results_dir, sub_item)
            metrics_path = os.path.join(folder, 'metrics.json')
            if not os.path.exists(metrics_path):
                print(f"Warning: {metrics_path} not found, skip.")
                continue
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            subitem_scores = {m: [] for m in metrics}
            # Collect per-sample scores
            for sample_id, sample in data.items():
                for m in metrics:
                    if m in sample and sample[m] is not None:
                        value = sample[m]
                        norm_value = normalize_score(value)
                        subitem_scores[m].append(norm_value)
                        type_all_scores[b_type][m].append(norm_value)
                        type_all_values_flat[b_type].append(norm_value)
            # Calculate per-subitem average, removing None
            subitem_avg = {}
            for m in score_name_order:
                if m in metrics:
                    vals = subitem_scores[m]
                    if vals:  # Only save if not empty
                        avg = sum(vals) / len(vals)
                        subitem_avg[score_name[m]] = avg
            if subitem_avg:  # Only add subitems with at least one valid score
                subitem_summary[b_type][sub_item] = subitem_avg

    # Calculate per-category average, remove None
    type_avg = {}
    for b_type in kris_benchamrk_type:  # Respect input order
        cat_avg = {}
        for m in score_name_order:
            if m == "average_score":
                all_vals = type_all_values_flat[b_type]
                if all_vals:
                    avg = sum(all_vals) / len(all_vals)
                    cat_avg[score_name[m]] = avg
            elif m in kris_benchamrk_metric[b_type]:
                vals = type_all_scores[b_type][m]
                if vals:
                    avg = sum(vals) / len(vals)
                    cat_avg[score_name[m]] = avg
        if cat_avg:  # Only add categories with at least one valid score
            type_avg[b_type] = cat_avg

    # Calculate meta-category averages (only AVG, as mean of all child category flat scores)
    meta_avg = {}
    for meta_cat, child_cats in meta_categories.items():
        vals = []
        for child in child_cats:
            vals.extend(type_all_values_flat[child])
        if vals:
            meta_avg[meta_cat] = {"AVG": sum(vals) / len(vals)}

    # Calculate Overall (mean of ALL scores)
    all_scores = []
    for b_type in kris_benchamrk_type:
        all_scores.extend(type_all_values_flat[b_type])
    overall = {}
    if all_scores:
        overall["AVG"] = sum(all_scores) / len(all_scores)

    return type_avg, subitem_summary, meta_avg, overall

def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark scores.")
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory containing subitem folders.')
    args = parser.parse_args()

    results_dir = args.results_dir
    type_result, subitem_result, meta_result, overall_result = summarize_benchmark_scores_with_normalization(results_dir, max_score=5)

    # Merge meta categories and overall into summary (after normal categories, in specified order)
    summary_with_meta = type_result.copy()
    for meta_cat in meta_category_order:
        if meta_cat in meta_result:
            summary_with_meta[meta_cat] = meta_result[meta_cat]
    if overall_result:
        summary_with_meta["Overall"] = overall_result

    # Prepare output dict (only non-None entries)
    final_result = {
        "summary": summary_with_meta,
        "subitems": subitem_result
    }

    # Write to JSON file
    result_json_path = os.path.join(results_dir, 'results.json')
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    # Print summary, order: original categories, meta categories, overall, only non-None
    print("Category, meta-category, and overall average scores (100-point scale):")
    output_order = list(kris_benchamrk_type.keys()) + meta_category_order + ["Overall"]
    for b_type in output_order:
        if b_type not in summary_with_meta:
            continue
        scores = summary_with_meta[b_type]
        # Meta and overall categories only print AVG
        if b_type in meta_categories or b_type == "Overall":
            if "AVG" in scores:
                val = scores["AVG"]
                print(f"{b_type}:")
                print(f"  AVG: {val:.2f}")
        else:
            print(f"{b_type}:")
            for _score_key in score_name_order:
                score_short = score_name[_score_key]
                if score_short in scores:
                    val = scores[score_short]
                    print(f"  {score_short}: {val:.2f}")

if __name__ == "__main__":
    main()
