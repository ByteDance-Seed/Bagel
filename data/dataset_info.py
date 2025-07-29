# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, EditJSONLIterableDataset, PiEditIterableDataset, PiEditAllViewsIterableDataset, PiRobotQAAllViewsIterableDataset,PiT2IAllViewsIterableDataset, PiTextOnlyIterableDataset,PiWebQAIterableDataset
from .t2i_dataset import T2IIterableDataset, T2IIWebDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'simple_edit': EditJSONLIterableDataset,
    'sharded_edit': EditJSONLIterableDataset,
    'pi_edit': PiEditIterableDataset,
    'pi_edit2': PiEditIterableDataset,
    'pi_edit3': PiEditIterableDataset,
    'pi_edit_allviews': PiEditAllViewsIterableDataset,
    'pi_edit_allviews1': PiEditAllViewsIterableDataset,
    'pi_edit_allviews2': PiEditAllViewsIterableDataset,
    'pi_edit_allviews3': PiEditAllViewsIterableDataset,
    'pi_edit_allviews4': PiEditAllViewsIterableDataset,
    'blip30_longcaption': T2IIWebDataset,
    'blip30_journeyDb': T2IIWebDataset,
    'blip30_shortcaption': T2IIWebDataset,
    'pi_robot_bbox_cap': PiRobotQAAllViewsIterableDataset,
    'pi_t2i_dataset': PiT2IAllViewsIterableDataset,
    'pi_textonly': PiTextOnlyIterableDataset,
    'pi_webqa_dataset': PiWebQAIterableDataset,
}


DATASET_INFO = {
    'blip30_longcaption': {
        'blip30_longcaption': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/BLIP3o-Pretrain-Long-Caption/', # path of the parquet files
            # 'num_files': 10, # number of data units to be sharded across all ranks and workers
            # 'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'blip30_journeyDb': {
        'blip30_journeyDb': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/BLIP3o-Pretrain-JourneyDB/', # path of the parquet files
            # 'num_files': 10, # number of data units to be sharded across all ranks and workers
            # 'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'blip30_shortcaption': {
        'blip30_shortcaption': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/BLIP3o-Pretrain-Short-Caption/', # path of the parquet files
            # 'num_files': 10, # number of data units to be sharded across all ranks and workers
            # 'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    't2i_pretrain': {
        't2i': {
            'data_dir': '/home/liliyu/workspace/BAGEL/example_data/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '/home/liliyu/workspace/BAGEL/example_data/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": '/home/liliyu/workspace/BAGEL/example_data/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
    },
    'sharded_edit':{
        'sharded_edit': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
            'json_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/annotations_GPT4V/', # information of the parquet files
		},

    },
    'simple_edit':{
        'simple_edit': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
            'json_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/annotations_GPT4V/', # information of the parquet files
		},
        'seed_part1': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
            'json_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/annotations_GPT4V/', # information of the parquet files
		},
    },
    "pi_edit": {
        'pi_edit': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
        'arx_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    "pi_edit2": {
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    "pi_edit3": {
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    "pi_edit_allviews": {
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    "pi_edit_allviews1": {
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    "pi_edit_allviews2": {
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    "pi_edit_allviews3": {
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    "pi_edit_allviews4": {
        'ur5_endspan_448': {
            'data_dir': '/mnt/weka/checkpoints/hf_data/SEED-Data-Edit-Part1-Openimages/auto_editing/openimages/images/',
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '/home/liliyu/workspace/BAGEL/example_data/bagel_example/vlm/images',
			'jsonl_path': '/home/liliyu/workspace/BAGEL/example_data/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}