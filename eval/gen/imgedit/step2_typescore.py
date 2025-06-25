# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import argparse
from collections import defaultdict

def compute_edit_type_averages(score_dict, meta_dict):
    edit_type_scores = defaultdict(list)
    all_scores = []

    for key, score in score_dict.items():
        meta = meta_dict.get(key, {})
        edit_type = meta.get("edit_type")
        if edit_type is not None:
            edit_type_scores[edit_type].append(score)
        all_scores.append(score)

    averaged_by_type = {
        etype: round(sum(scores) / len(scores), 2)
        for etype, scores in edit_type_scores.items() if scores
    }
    if all_scores:
        averaged_by_type['overall'] = round(sum(all_scores) / len(all_scores), 2)

    return averaged_by_type

def main():
    parser = argparse.ArgumentParser(description="Calculate edit type averages")
    parser.add_argument('--average_score_json', type=str, required=True, help='path to the JSON file containing the scores')
    parser.add_argument('--edit_json', type=str, required=True, help='Path  to the JSON file containing the basic edit information')
    parser.add_argument('--typescore_json', type=str, required=True, help='Path  to the JSON file containing the edit type scores')

    args = parser.parse_args()

    with open(args.average_score_json, 'r', encoding='utf-8') as f:
        score_data = json.load(f)

    with open(args.edit_json, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    averaged_result = compute_edit_type_averages(score_data, meta_data)
    for k, v in averaged_result.items():
        print(f"{k}: {v}")

    with open(args.typescore_json, 'w', encoding='utf-8') as f:
        json.dump(averaged_result, f, indent=2)


if __name__ == '__main__':
    main()