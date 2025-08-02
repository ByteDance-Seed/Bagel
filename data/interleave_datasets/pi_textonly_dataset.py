# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
This script demonstrates how to use PyTorch's DataLoader with a our datasets.

# If you are using this in a separate repo, make sure to do the following steps:
#  export MONOPI_REPO=/home/liliyu/workspace/monopi
#  pip install --requirement $MONOPI_REPO/monopi/model/data/requirements.txt
#  pip install -e $MONOPI_REPO
"""
import logging
import json
import os
import traceback
import io
import random
from PIL import Image, ImageFile, PngImagePlugin
# gazelle:ignore torch
import torch

from monopi.model.configs import registered_configs as register_cfg
from monopi.lib.py.image import image as lib_image
import monopi.lib.py.ml.jax.string_encode as string_encode

import wandb
import getpass
import dataclasses
from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
import numpy as np
import cloudpickle
import multiprocessing as mp
import time
import re
from .pi_data_utils import convert_loc_to_bbox, create_pi_dataset_old, create_pi_dataset

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class PiTextOnlyIterableDataset(InterleavedBaseIterableDataset):
    def __init__(
        self, pi_config_name, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, experiment_name='debug', 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0, n_log_examples=100, image_keys="image_0,image_2",   
        training_text_loss=False, with_condition=False, force_drop_all_prob=0.15, rank0_only=False, use_vit_as_condition=False
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.pi_config = pi_config_name
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.data_status = data_status
        self.rank0_only = rank0_only
        self.use_vit_as_condition = use_vit_as_condition
        self.data_paths = self.get_data_paths(local_rank, world_size)
        self.experiment_name = experiment_name

        self.data_table = wandb.Table(columns=["id", "image", "instruction", "target"])
        self.n_log_examples = n_log_examples
        self.image_key_list = [key.strip() for key in image_keys.split(',')]
        self.training_text_loss = training_text_loss
        self.with_condition = with_condition
        self.force_drop_all_prob = force_drop_all_prob
        self.set_epoch()


    def get_data_paths(self, local_rank, world_size):
        config = register_cfg.get_config(self.pi_config)
        if self.rank0_only:
            data_paths = create_pi_dataset(config, split="train", local_rank=local_rank, world_size=world_size)
        else:
            data_paths = create_pi_dataset_old(config, split="train", local_rank=local_rank, world_size=world_size)
        return data_paths


    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}" 
        )

        while True:
            frame_idx = 0
            # Skip the rows that have already been trained
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, row in enumerate(data_paths_per_worker_, start=row_start_id):
                data = self._init_data()
                try:
                    json_data = json.loads(str(string_encode.decode_str(row['raw_text'])))
                    
                    prompt = "Question: " + json_data["question"] + "\nAnswer: " + json_data["answer"]
                    data = self._add_text(data, prompt, need_loss=True)
                except Exception as e:
                    print(f"Error adding text: {e}")
                    continue
                                
                if len(data) == 0:
                    continue
                # Add logger for debugging
                if row_idx <= self.n_log_examples:
                    # Create side-by-side full_example with text
                    print(f"row_idx: {row_idx}, prompt: {prompt}")
                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield data

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
