# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import argparse
import os
import os.path as osp
from typing import Callable, Iterable
import time
import re
import openai
import pandas as pd
from utils import *

openai.api_key = os.getenv('OPENAI_API_KEY')

subtask_dic = {
    "Temp": [
        "Life Progression",
        "Material Progression",
        "Environmental Cycles",
        "Societal Transformation",
    ],
    "Causal": [
        "Structural Deformation",
        "State Transition",
        "Chemical and Biological Transformation",
        "Physics Manifestation",
    ],
    "Spa": [
        "Component Assembly",
        "Object Arrangement",
        "Viewpoint Generation",
        "Structural Inference",
        "Layout Reasoning",
    ],
    "Logic": ["Pattern Prediction", "Mathematical Derivation", "Puzzle Solving"],
}

def gpt_generate(inputs, temperature=0, max_tokens=4096, image_size=768, **kwargs):
    input_msgs = prepare_inputs(inputs, image_size=image_size)
    temperature = kwargs.pop('temperature', temperature)
    max_tokens = kwargs.pop('max_tokens', max_tokens)
    retries = 5
    for attempt in range(1, retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                stream=False,
                messages=input_msgs,
                max_tokens=max_tokens,
                n=1,
                temperature=temperature,
            )
            response = response.to_dict()
            break
        except Exception as e:
            print(f'{inputs}')
            print(f"❌ [Attempt {attempt}/{retries}] Unexpected error: {e}")
            if attempt==retries:
                raise e
            time.sleep(3)

    ret_code = getattr(response, "status_code", 0) or 0
    ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

    answer = 'Failed to obtain answer via API. '
    try:
        answer = response['choices'][0]['message']['content'].strip()
    except Exception as err:
        print(f'{type(err)}: {err}')
        print(response)
    return ret_code, answer, response

def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    exception = futures[idx].exception()
                    if exception is not None:
                        raise exception
                    else:
                        results[idx] = futures[idx].result()
                        new_finished.add(idx)
                        if keys is not None:
                            res[keys[idx]] = results[idx]
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            time.sleep(0.1)
        pbar.close()

    if save is not None:
        dump(res, save)
    return results

def find_image(output_dir, index):
    for suffix in ['png', 'jpg', 'jpeg']:
        img_path = osp.join(output_dir, f"{index}.{suffix}")
        if osp.exists(img_path):
            return img_path
    raise FileNotFoundError(f"Cannot find output images {index} in {output_dir}!!!")

def eval_vanilla(item, input_dir, output_dir, **kwargs):
    instruct = item['instruction']
    index = item['index']
    category = item['category']
    output_dir = osp.join(output_dir, category)
    img2 = find_image(output_dir, index)
    judge_exist = item.get('judge', None)
    judge_rea_require_img = False
    
    if category in ['temporal_reasoning', 'causal_reasoning']:
        img1 = osp.join(input_dir, item['image'])
        reference = item['reference']
        if "reference_img" in item and not pd.isna(item['reasoning_img']):
            judge_rea_require_img = True
            prompt_rea = prompt_reasoning_w_input.format(instruct=instruct, reference=reference)
        else:
            prompt_rea = prompt_reasoning.format(instruct=instruct, reference=reference)

        prompt_cons = prompt_consist.format(instruct=instruct)
        prompt_qua = prompt_generation

    elif category == 'spatial_reasoning':
        img1 = osp.join(input_dir, item['image'])
        if "reference_img" in item and not pd.isna(item['reference_img']):
            judge_rea_require_img = True
            img1 = osp.join(input_dir, item['reference_img'])
            prompt_rea = prompt_spatial_ref_img.format(instruct=instruct)
        elif not pd.isna(item['reasoning_img']):
            judge_rea_require_img = True
            reference = item['reference']
            prompt_rea = prompt_spatial_ref_w_input.format(instruct=instruct, reference=reference)
        else:
            reference = item['reference']
            prompt_rea = prompt_spatial_ref.format(instruct=instruct, reference=reference)

        prompt_cons = prompt_spatial_cons.format(instruct=instruct)
        prompt_qua = prompt_spatial_qual

    elif category == 'logical_reasoning':
        if "reference_txt" in item and not pd.isna(item['reference_txt']):
            img1 = osp.join(input_dir, item['image'])
            reference = item['reference_txt']
            prompt_cons = prompt_logical_cons_ans.format(instruct=instruct, reference=reference)
            prompt_rea = prompt_logical_txt.format(instruct=instruct, reference=reference)
        elif "reference_img" in item and not pd.isna(item['reference_img']):
            judge_rea_require_img=True
            img1 = osp.join(input_dir, item['reference_img'])
            prompt_cons = prompt_logical_cons.format(instruct=instruct)
            if 'reasoning_wo_ins' in item:
                prompt_rea = prompt_logical_img_wo_q
            else:
                prompt_rea = prompt_logical_img.format(instruct=instruct)

    if 'consistency_free' in item and not pd.isna(item['consistency_free']):
        consist_judge = None
        print('Consistency Judgement not required. Ignore.')
    else:
        if judge_exist and 'judge1' in judge_exist:
            consist_judge = judge_exist['judge1']
        else:
            message = []
            text = {'type': 'text', 'value': prompt_cons}
            image1 = {
                'type': 'image',
                'value': img1,
            }
            image2 = {
                'type': 'image',
                'value': img2,
            }
            message.append(text)
            message.append(image1)
            message.append(image2)
            print(message)

            ret_code, consist_judge, response = gpt_generate(message, **kwargs)

    if judge_exist and 'judge2' in judge_exist:
        answer2 = judge_exist['judge2']
    else:
        if judge_rea_require_img:
            message2 = [
                {'type': 'text', 'value': prompt_rea}, 
                {'type': 'image','value': img1},
                {'type': 'image','value': img2}
                ]
        else:
            message2 = [{'type': 'text', 'value': prompt_rea}, {
                'type': 'image',
                'value': img2,
            }]
        print(message2)

        ret_code2, answer2, response2 = gpt_generate(message2)

    if category in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        if judge_exist and 'judge3' in judge_exist:
            answer3 = judge_exist['judge3']
        else:
            message3 = [{'type': 'text', 'value': prompt_qua}, {
                'type': 'image',
                'value': img2,
            }]
            ret_code3, answer3, response3 = gpt_generate(message3)

        return dict(judge1=consist_judge, judge2=answer2, judge3=answer3)
    else:
        return dict(judge1=consist_judge, judge2=answer2)
    return dict(judge1=consist_judge)


def extract(answer):
    matches = re.findall(r'\*?\*?Final Score\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers != []:
            return numbers

    matches = re.findall(r'\*?\*?Final Scores\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        return numbers
    else:
        return None

def calculate_score(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        if 'consistency_free' in row and row['consistency_free']:
            score = 0.2 * row['VisualPlausibility'] + 0.8 * row['Reasoning']
        else:
            score = 0.3 * row['ApprConsistency'] + 0.5 * row['Reasoning'] + 0.2 * row['VisualPlausibility']
        
    elif row['category'] == 'logical_reasoning':
        score = 0.3 * row['ApprConsistency'] + 0.7 * row['Reasoning']
    if row['Reasoning'] == 1:
        score = score * 0.5
        score = 1 if score<1 else score
    return score

def calculate_completion(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        return (
            1
            if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 and row['VisualPlausibility'] == 5
            else 0
        )
    elif row['category']=='logical_reasoning':
        return (
            1 if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 else 0
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Json Path')
    parser.add_argument('--output', type=str, required=True, help='Output Image Dir, outputs/MODEL_NAME')
    parser.add_argument('--input', type=str, default='data', help='Input Image Dir')
    parser.add_argument('--prefix', type=str, default=None, help='output json prefix')
    parser.add_argument('--model', type=str, default=None, help='Model Name')
    parser.add_argument('--nproc', type=int, default=4, help='n processes for api')

    args = parser.parse_args()

    model_name = args.output.split('/')[-1] if args.model is None else args.model
    if not args.prefix:
        tmp_file = f"{args.output}/{model_name}.pkl"
        judge_res = f"{args.output}/{model_name}_judge.xlsx"
        score_file = f"{args.output}/{model_name}_judge.csv"
    else:
        tmp_file = f"{args.output}/{args.prefix}_{model_name}.pkl"
        judge_res = f"{args.output}/{args.prefix}_{model_name}_judge.xlsx"
        score_file = f"{args.output}/{args.prefix}_{model_name}_judge.csv"

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    data = json.load(open(args.data))
    data = pd.DataFrame(data)

    result = {}
    if osp.exists(tmp_file):
        result = load(tmp_file)

    items = []

    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(item=x, input_dir=args.input, output_dir=args.output) for x in items]
    keys = [x['index'] for x in items]
    # breakpoint()
    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=args.nproc, chunksize=args.nproc, save=tmp_file, keys=keys)
        result = load(tmp_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v

    judges = [result[i] for i in data['index']]

    scores, judge_combine, judge_cons, judge_reas, judge_qua = [], [], [], [], []

    for judge in judges:
        if judge['judge1'] is None:
            judge_combine.append(
                'REASONING\n\n'
                + judge['judge2']
                + '\n\nQUALITY\n\n'
                + judge['judge3']
            )
            judge_cons.append(None)
            judge_reas.append(judge['judge2'])
            judge_qua.append(judge['judge3'])

            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score2 or not score3:
                score=None
            else:
                score = [None]+score2+score3
        elif 'judge3' not in judge:
            judge_combine.append(
                'CONSISTENCY\n\n'
                + judge['judge1']
                + '\n\nREASONING\n\n'
                + judge['judge2']
            )
            judge_cons.append(judge['judge1'])
            judge_reas.append(judge['judge2'])
            judge_qua.append(None)

            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            if not score1 or not score2:
                score=None
            else:
                score = score1+score2
        elif 'judge2' not in judge:
            judge_combine.append(judge['judge1'])
            score = [extract(judge['judge1'])[1], extract(judge['judge1'])[0]]
        else:
            try:
                judge_combine.append(
                    'CONSISTENCY\n\n'
                    + judge['judge1']
                    + '\n\nREASONING\n\n'
                    + judge['judge2']
                    + '\n\nQUALITY\n\n'
                    + judge['judge3']
                )
                judge_cons.append(judge['judge1'])
                judge_reas.append(judge['judge2'])
                judge_qua.append(judge['judge3'])
            except Exception as e:
                print(e)
                breakpoint()
            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score1 or not score2 or not score3:
                score=None
            else:
                score = score1+score2+score3
        scores.append(score)

    reasoning = []
    img_consist = []
    gen_quality = []
    match_log = []

    for score in scores:
        if score:
            match_log.append('succeed')
            if len(score)==3:
                img_consist.append(score[0])
                reasoning.append(score[1])
                gen_quality.append(score[2])

            elif len(score)==2:
                reasoning.append(4 * min(score[1], 1) + 1)
                img_consist.append(4 * min(score[0], 1) + 1)
                gen_quality.append(None)
        else:
            img_consist.append(None)
            reasoning.append(None)
            gen_quality.append(None)
            match_log.append('failed')
    # breakpoint()
    data['Reasoning'] = reasoning
    data['ApprConsistency'] = img_consist
    data['VisualPlausibility'] = gen_quality
    data['match_log'] = match_log
    # data['judge'] = judge_combine
    data['judge_cons'] = judge_cons
    data['judge_reas'] = judge_reas
    data['judge_qua'] = judge_qua

    data['score'] = data.apply(calculate_score, axis=1)
    data['complete'] = data.apply(calculate_completion, axis=1)

    dump(data, judge_res)

    df_causal = data[data['category'] == 'causal_reasoning']
    df_temporal = data[data['category'] == 'temporal_reasoning']
    df_spatial = data[data['category'] == 'spatial_reasoning']
    df_logical = data[data['category'] == 'logical_reasoning']

    score_final = data['score'].mean()
    completion_rate = data['complete'].mean()
    
    # calculate score and accuracy per main task
    temporal_final, temporal_comp_rate = df_temporal['score'].mean(), df_temporal['complete'].mean()
    causal_final, causal_comp_rate = df_causal['score'].mean(), df_causal['complete'].mean()
    spatial_final, spatial_comp_rate = df_spatial['score'].mean(), df_spatial['complete'].mean()
    logical_final, logical_comp_rate = df_logical['score'].mean(), df_logical['complete'].mean()

    reasoning_average = data['Reasoning'].mean()
    img_consist_average = data['ApprConsistency'].mean()
    generation_quality = data['VisualPlausibility'].mean()

    temp_rea_avg, temp_cons_avg, temp_qua_avg = df_temporal['Reasoning'].mean(), df_temporal['ApprConsistency'].mean(), df_temporal['VisualPlausibility'].mean()
    cau_rea_avg, cau_cons_avg, cau_qua_avg = df_causal['Reasoning'].mean(), df_causal['ApprConsistency'].mean(), df_causal['VisualPlausibility'].mean()
    spa_rea_avg, spa_cons_avg, spa_qua_avg = df_spatial['Reasoning'].mean(), df_spatial['ApprConsistency'].mean(), df_spatial['VisualPlausibility'].mean()
    logic_rea_avg, logic_cons_avg, logic_qua_avg = df_logical['Reasoning'].mean(), df_logical['ApprConsistency'].mean(), df_logical['VisualPlausibility'].mean()

    def trans_to_percent(s):
        return 25*(s-1)
    
    # calculate score and accuracy per subtask
    average_scores_by_subtask = data.groupby('subtask')['score'].mean()
    average_acc_by_subtask = data.groupby('subtask')['complete'].mean()

    average_scores_dict = average_scores_by_subtask.to_dict()
    average_acc_dict = average_acc_by_subtask.to_dict()
    
    subtask_results = {}
    for k, v in average_scores_dict.items():
        subtask_results[k] = [v, trans_to_percent(v), average_acc_dict[k]]
    
    sorted_subtask_results = {}
    for main_task_prefix, subtasks in subtask_dic.items():
        for subtask in subtasks:
            if subtask in subtask_results:
                new_key = f"{main_task_prefix}-{subtask}"
                sorted_subtask_results[new_key] = subtask_results[subtask]

    final_score = dict(
        Overall=[score_final, trans_to_percent(score_final), completion_rate],
        Temporal=[temporal_final, trans_to_percent(temporal_final), temporal_comp_rate],
        Causal=[causal_final, trans_to_percent(causal_final), causal_comp_rate],
        Spatial=[spatial_final, trans_to_percent(spatial_final), spatial_comp_rate],
        Logical=[logical_final, trans_to_percent(logical_final), logical_comp_rate],
        Overall_Reasoning=[reasoning_average, trans_to_percent(reasoning_average), None],
        Overall_ApprConsistency=[img_consist_average, trans_to_percent(img_consist_average), None],
        Overall_VisualPlausibility_total=[generation_quality, trans_to_percent(generation_quality), None],
        Temporal_Reasoning = [temp_rea_avg, trans_to_percent(temp_rea_avg), None],
        Temporal_Consistency = [temp_cons_avg, trans_to_percent(temp_cons_avg), None],
        Temporal_Quality = [temp_qua_avg, trans_to_percent(temp_qua_avg), None],
        Causal_Reasoning = [cau_rea_avg, trans_to_percent(cau_rea_avg), None],
        Causal_Consistency = [cau_cons_avg, trans_to_percent(cau_cons_avg), None],
        Causal_Quality = [cau_qua_avg, trans_to_percent(cau_qua_avg), None],
        Spatial_Reasoning = [spa_rea_avg, trans_to_percent(spa_rea_avg), None],
        Spatial_Consistency = [spa_cons_avg, trans_to_percent(spa_cons_avg), None],
        Spatial_Quality = [spa_qua_avg, trans_to_percent(spa_qua_avg), None],
        Logical_Reasoning = [logic_rea_avg, trans_to_percent(logic_rea_avg), None],
        Logical_Consistency = [logic_cons_avg, trans_to_percent(logic_cons_avg), None],
        **sorted_subtask_results
    )

    df = pd.DataFrame(final_score, index=["Score-Origin", "Score-Percentage", "Accuracy"]).T
    df.reset_index(inplace=True)
    df.columns = ["-", "Score-Origin", "Score-Percentage", "Accuracy"]
    df.to_csv(score_file, index=False)
    print(df)


if __name__ == '__main__':
    base_url = "your_api_url"
    api_version = "2024-03-01-preview"
    api_key = openai.api_key
    model_name = "gpt-4.1-2025-04-14"
    openai_client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=api_key,
    )
    main()
