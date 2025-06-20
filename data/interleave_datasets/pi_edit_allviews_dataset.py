# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
This script demonstrates how to use PyTorch's DataLoader with a our datasets.

# If you are using this in a separate repo, make sure to do the following steps:
#  export MONOPI_REPO=/home/liliyu/workspace/monopi
#  pip install --requirement $MONOPI_REPO/monopi/model/data/requirements.txt
#  pip install -e $MONOPI_REPO
"""
import json
import os
import traceback
import io
import random
from PIL import Image, ImageFile, PngImagePlugin
# gazelle:ignore torch
import torch
from monopi.model.configs import config as _config
from monopi.model.configs import registered_configs as register_cfg
from monopi.model.data import dataloader
from monopi.experimental.dibyaghosh import utils as experimental_utils
from monopi.model.configs import registered_configs as register_cfg
from monopi.lib.py.image import image as lib_image
import monopi.lib.py.ml.jax.string_encode as string_encode
import wandb
import getpass
from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def create_pi_dataset(
    config: _config.TrainConfig, *, split: str = "train", num_epochs: int = 1
):
    """Creates a PyTorch dataset from a config name."""
    config.data.return_compressed_images = True
    # create an dataset
    # experimental_utils.cache_specs(
    #     config.data.task_mixture_config, f"/home/{getpass.getuser()}/cached_specs"
    # )

    task_mixture = dataloader.create_task_mixture(config.data)
    mixture = task_mixture.mixtures[split]
    dt = mixture.get_dataset(num_epochs=num_epochs, shuffle=True)
    return dt

class PiEditAllViewsIterableDataset(InterleavedBaseIterableDataset):
    def __init__(
        self, pi_config_name, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, experiment_name='debug', 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0, n_log_examples=100, image_key="image_0",   
        indepent_image_modeling=False,
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
        self.data_paths = self.get_data_paths()
        self.experiment_name = experiment_name

        self.data_table = wandb.Table(columns=["id", "image", "instruction", "target"])
        self.n_log_examples = n_log_examples
        self.image_key = image_key
        self.set_epoch()


    def get_data_paths(self):
        config = register_cfg.get_config(self.pi_config)
        data_paths = create_pi_dataset(config, split="train")
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
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, row in enumerate(data_paths_per_worker_, start=row_start_id):
                data = self._init_data()
                for image_key in ["image_0", "image_2", "image_3"]:
                    condition_image = row["image"][self.image_key]
                    condition_image = Image.fromarray(lib_image.decompress_image_if_needed(condition_image))
                    data = self._add_image(
                        data, 
                        condition_image,
                        need_loss=False, 
                        need_vae=True, 
                        need_vit=True, 
                    )
                edit_instruction = str(string_encode.decode_str(row["raw_text"]))
                data = self._add_text(data, edit_instruction, need_loss=False)
                for image_key in ["image_0", "image_2", "image_3"]:
                    edited_image = row["future_image"][f'future_{self.image_key}']
                    edited_image = Image.fromarray(lib_image.decompress_image_if_needed(edited_image))
                    data = self._add_image(
                        data, 
                        edited_image,
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )

                if len(data) == 0:
                    continue
                # Add logger for debugging
                if row_idx <= self.n_log_examples:
                    # Create side-by-side full_example with text
                    self.save_example_image(condition_image, edited_image, edit_instruction, row_idx)
                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield data

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
