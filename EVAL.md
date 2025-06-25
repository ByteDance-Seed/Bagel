# VLM
We follow [InternVL2](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html) to evaluate the performance on MME, MMBench, MMMU, MMVet, MathVista and MMVP.

## Data prepration
Please follow the [InternVL2](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html) to prepare the corresponding data. And the link the data under `vlm`.

The final directory structure is:
```shell
data
├── MathVista
├── mmbench
├── mme
├── MMMU
├── mm-vet
└── MMVP
```

## Evaluation

Directly run `scripts/eval/run_eval_vlm.sh` to evaluate different benchmarks. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Increase `GPUS` if you want to run faster.
- For MMBench, please use the official [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).
- For MMVet, please use the official [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator).
- For MathVista, please set `$openai_api_key` in `scripts/eval/run_eval_vlm.sh` and `your_api_url` in `eval/vlm/eval/mathvista/utilities.py`. The default GPT version is `gpt-4o-2024-11-20`.
- For MMMU, we use CoT in the report, which improve the accuracy by about 2%. For evaluation of the oprn-ended answer, we use GPT-4o for judgement.


# GenEval
We modify the code in [GenEval](https://github.com/djghosh13/geneval/tree/main) for faster evaluation.

## Setup
Install the following dependencies:
```shell
pip install open-clip-torch
pip install clip-benchmark
pip install --upgrade setuptools

sudo pip install -U openmim
sudo mim install mmengine mmcv-full==1.7.2

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

Download Detector:
```shell
cd ./eval/gen/geneval
mkdir model

bash ./evaluation/download_models.sh ./model
```

## Evaluation
Directly run `scripts/eval/run_geneval.sh` to evaluate GenEVAL. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `metadata_file` to `./eval/gen/geneval/prompts/evaluation_metadata.jsonl` for original GenEval prompts.


# WISE
We modify the code in [WISE](https://github.com/PKU-YuanGroup/WISE/tree/main) for faster evaluation.


## Evaluation
Directly run `scripts/eval/run_wise.sh` to evaluate WISE. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `$openai_api_key` in `scripts/eval/run_wise.sh` and `your_api_url` in `eval/gen/wise/gpt_eval_mp.py`. The default GPT version is `gpt-4o-2024-11-20`.
- Use `think` for thinking mode.



# GEdit-Bench
We adopt the code in [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md) for evaluation.

## Evaluation

Modify the model path, the output path, the api key, and the api url in `scripts/eval/run_gedit.sh`. Then, run the following command:
```shell
bash script/eval/run_gedit.sh
```
The GPT version for evaluation is `gpt-4.1-2025-04-14`.


# IntelligentBench
TBD


# KRIS
We modify the code in [KRIS-Bench](https://github.com/mercurystraw/Kris_Bench) for faster evaluation.

## Data prepration
Please download the benchmark data from [KRIS-Bench](https://huggingface.co/datasets/Liang0223/KRIS_Bench) and and place it in the `KRIS_Bench` directory.

The final directory structure is:
```shell
KRIS_Bench
├── abstract_reasoning
├── anomaly_correction
├── biology
├── chemistry
├── color_change
├── count_change
├── geography
├── humanities
├── mathematics
├── medicine
├── multi-element_composition
├── multi-instruction_execution
├── part_completion
├── physics
├── position_movement
├── practical_knowledge
├── rule-based_reasoning
├── size_adjustment
├── temporal_prediction
└── viewpoint_change
```

## Evaluation
Directly run `scripts/eval/run_kris.sh` to evaluate KRIS-Bench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `$openai_api_key` in `scripts/eval/run_kris.sh` and `your_api_url` in `eval/gen/kris/metrics_xx.py`. The default GPT version is `gpt-4o-2024-11-20`.
- Use `think` for thinking mode.
- We set `cfg_text_scale=4` and `cfg_img_scale=1.5` by default. Additionally, `cfg_renorm_min=0` is specified for CFG Renorm.

<details>
<summary><b>Results</b></summary>
<pre>
Category, meta-category, and overall average scores (100-point scale):
Attribute Perception:
  VC: 76.64
  VQ: 74.45
  IF: 41.73
  AVG: 64.27
Spatial Perception:
  VC: 70.25
  VQ: 80.00
  IF: 37.00
  AVG: 62.42
Temporal Prediction:
  VC: 36.49
  VQ: 61.82
  IF: 29.05
  AVG: 42.45
Social Science:
  VC: 76.20
  VQ: 78.80
  IF: 37.00
  KP: 29.60
  AVG: 55.40
Natural Science:
  VC: 69.59
  VQ: 84.03
  IF: 40.27
  KP: 30.15
  AVG: 56.01
Logical Reasoning:
  VC: 80.17
  VQ: 85.67
  IF: 26.33
  KP: 18.00
  AVG: 52.54
Instruction Decomposition:
  VC: 40.17
  VQ: 69.50
  IF: 42.00
  AVG: 50.56
Factual Knowledge:
  AVG: 60.26
Conceptual Knowledge:
  AVG: 55.86
Procedural Knowledge:
  AVG: 51.69
Overall:
  AVG: 56.21
</pre>
</details>

<details>
<summary><b>Results w/ CoT</b></summary>
<pre>
Category, meta-category, and overall average scores (100-point scale):
Attribute Perception:
  VC: 75.09
  VQ: 74.00
  IF: 53.18
  AVG: 67.42
Spatial Perception:
  VC: 78.75
  VQ: 87.25
  IF: 39.00
  AVG: 68.33
Temporal Prediction:
  VC: 48.31
  VQ: 81.08
  IF: 46.62
  AVG: 58.67
Social Science:
  VC: 80.40
  VQ: 79.40
  IF: 51.60
  KP: 42.80
  AVG: 63.55
Natural Science:
  VC: 67.68
  VQ: 82.95
  IF: 52.10
  KP: 42.88
  AVG: 61.40
Logical Reasoning:
  VC: 62.83
  VQ: 79.67
  IF: 28.33
  KP: 21.67
  AVG: 48.12
Instruction Decomposition:
  VC: 47.83
  VQ: 66.83
  IF: 36.00
  AVG: 50.22
Factual Knowledge:
  AVG: 66.18
Conceptual Knowledge:
  AVG: 61.92
Procedural Knowledge:
  AVG: 49.02
Overall:
  AVG: 60.18
</pre>
</details>


# RISE
We modify the code in [RISEBench](https://github.com/PhoenixZ810/RISEBench) for faster evaluation.

## Data prepration
Please download the benchmark data from [RISEBench](https://huggingface.co/datasets/PhoenixZ/RISEBench) and and place it in the `data` directory.

The final directory structure is:
```shell
data
├── datav2_total_w_subtask.json
├── causal_reasoning_images
├── logical_reasoning_images
├── spatial_reasoning_images
└── temporal_reasoning_images
```

## Evaluation
Directly run `scripts/eval/run_rise.sh` to evaluate RISEBench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `$openai_api_key` in `scripts/eval/run_rise.sh` and `your_api_url` in `eval/gen/rise/gpt_eval.py`. The default GPT version is `gpt-4.1-2025-04-14`.
- Use `think` for thinking mode.
- We set `cfg_text_scale=4` and `cfg_img_scale=2.0` by default. Additionally, `cfg_renorm_min=0` is specified for CFG Renorm.

<details>
<summary><b>Results (cfg_img_scale=1.5)</b></summary>
<pre>
                                                -  Score-Origin  Score-Percentage  Accuracy
0                                         Overall      2.537778         38.444444  0.061111
1                                        Temporal      2.654118         41.352941  0.023529
2                                          Causal      2.788889         44.722222  0.055556
3                                         Spatial      3.452000         61.300000  0.140000
4                                         Logical      1.080000          2.000000  0.011765
5                               Overall_Reasoning      2.458333         36.458333       NaN
6                         Overall_ApprConsistency      3.141643         53.541076       NaN
7                Overall_VisualPlausibility_total      3.920000         73.000000       NaN
8                              Temporal_Reasoning      2.588235         39.705882       NaN
9                            Temporal_Consistency      3.250000         56.250000       NaN
10                               Temporal_Quality      3.505882         62.647059       NaN
11                               Causal_Reasoning      2.733333         43.333333       NaN
12                             Causal_Consistency      3.579545         64.488636       NaN
13                                 Causal_Quality      3.688889         67.222222       NaN
14                              Spatial_Reasoning      3.300000         57.500000       NaN
15                            Spatial_Consistency      3.330000         58.250000       NaN
16                                Spatial_Quality      4.480000         87.000000       NaN
17                              Logical_Reasoning      1.047059          1.176471       NaN
18                            Logical_Consistency      2.364706         34.117647       NaN
19                          Temp-Life Progression      2.757895         43.947368  0.000000
20                      Temp-Material Progression      2.500000         37.500000  0.021739
21                      Temp-Environmental Cycles      3.061538         51.538462  0.076923
22                   Temp-Societal Transformation      2.628571         40.714286  0.000000
23                  Causal-Structural Deformation      2.766667         44.166667  0.055556
24                        Causal-State Transition      3.112000         52.800000  0.080000
25  Causal-Chemical and Biological Transformation      2.325000         33.125000  0.062500
26                   Causal-Physics Manifestation      2.800000         45.000000  0.000000
27                         Spa-Component Assembly      3.434783         60.869565  0.043478
28                         Spa-Object Arrangement      2.733333         43.333333  0.000000
29                       Spa-Viewpoint Generation      3.629630         65.740741  0.222222
30                       Spa-Structural Inference      4.066667         76.666667  0.133333
31                           Spa-Layout Reasoning      3.234783         55.869565  0.217391
32                       Logic-Pattern Prediction      1.035484          0.887097  0.000000
33                  Logic-Mathematical Derivation      1.350000          8.750000  0.071429
34                           Logic-Puzzle Solving      1.020000          0.500000  0.000000
</pre>
</details>

<details>
<summary><b>Results w/ CoT</b></summary>
<pre>
                                                -  Score-Origin  Score-Percentage  Accuracy
0                                         Overall      2.933333         48.333333  0.119444
1                                        Temporal      3.336471         58.411765  0.058824
2                                          Causal      3.608889         65.222222  0.177778
3                                         Spatial      3.492000         62.300000  0.210000
4                                         Logical      1.157647          3.941176  0.011765
5                               Overall_Reasoning      2.836111         45.902778       NaN
6                         Overall_ApprConsistency      3.951841         73.796034       NaN
7                Overall_VisualPlausibility_total      4.203636         80.090909       NaN
8                              Temporal_Reasoning      3.188235         54.705882       NaN
9                            Temporal_Consistency      4.225000         80.625000       NaN
10                               Temporal_Quality      4.200000         80.000000       NaN
11                               Causal_Reasoning      3.533333         63.333333       NaN
12                             Causal_Consistency      4.386364         84.659091       NaN
13                                 Causal_Quality      4.100000         77.500000       NaN
14                              Spatial_Reasoning      3.350000         58.750000       NaN
15                            Spatial_Consistency      4.300000         82.500000       NaN
16                                Spatial_Quality      4.300000         82.500000       NaN
17                              Logical_Reasoning      1.141176          3.529412       NaN
18                            Logical_Consistency      2.835294         45.882353       NaN
19                          Temp-Life Progression      3.526316         63.157895  0.052632
20                      Temp-Material Progression      3.208696         55.217391  0.086957
21                      Temp-Environmental Cycles      3.584615         64.615385  0.000000
22                   Temp-Societal Transformation      3.200000         55.000000  0.000000
23                  Causal-Structural Deformation      3.750000         68.750000  0.138889
24                        Causal-State Transition      3.792000         69.800000  0.320000
25  Causal-Chemical and Biological Transformation      3.512500         62.812500  0.062500
26                   Causal-Physics Manifestation      2.984615         49.615385  0.153846
27                         Spa-Component Assembly      3.652174         66.304348  0.304348
28                         Spa-Object Arrangement      2.700000         42.500000  0.000000
29                       Spa-Viewpoint Generation      3.800000         70.000000  0.259259
30                       Spa-Structural Inference      3.680000         67.000000  0.266667
31                           Spa-Layout Reasoning      3.260870         56.521739  0.130435
32                       Logic-Pattern Prediction      1.064516          1.612903  0.000000
33                  Logic-Mathematical Derivation      1.707143         17.678571  0.071429
34                           Logic-Puzzle Solving      1.037500          0.937500  0.000000
</pre>
</details>


# ImgEdit
We modify the code in [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) for faster evaluation.

## Data prepration
Please download the benchmark data from [ImgEdit-Bench](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar) and and place it in the `Benchmark` directory.

The final directory structure is:
```shell
Benchmark
├── hard
├── multiturn
└── singleturn
    ├── judge_prompt.json
    ├── singleturn.json
    ├── animal
    ├── architecture
    ├── clothes
    ├── compose
    ├── daily object
    ├── for_add
    ├── human
    ├── style
    └── transport
```

## Evaluation
Directly run `scripts/eval/run_imgedit.sh` to evaluate ImgEdit-Bench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `$openai_api_key` in `scripts/eval/run_imgedit.sh` and `your_api_url` in `eval/gen/imgedit/basic_bench.py`. The default GPT version is `gpt-4o-2024-11-20`.
- We set `cfg_text_scale=4` and `cfg_img_scale=1.5` by default. Additionally, `cfg_renorm_min=0` is specified for CFG Renorm.

<details>
<summary><b>Results</b></summary>
<pre>
background: 3.28
adjust: 3.23
style: 4.26
extract: 1.48
remove: 2.99
add: 3.45
replace: 3.76
compose: 3.18
action: 4.38
overall: 3.28
</pre>
</details>