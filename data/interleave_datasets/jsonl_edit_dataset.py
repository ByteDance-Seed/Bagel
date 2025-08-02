# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import traceback
import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
import wandb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class EditJSONLIterableDataset(InterleavedBaseIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, json_dir_list,
        jsonl_path_list=[], data_dir_list=[], num_used_data=[], experiment_name='debug',
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=True, shuffle_seed=0, n_log_examples=100,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.data_status = data_status
        self.experiment_name = experiment_name
        self.data_table = wandb.Table(columns=["id", "image", "instruction", "target"])

        self.n_log_examples = n_log_examples
        self.data_paths = self.get_data_paths(
            json_dir_list, # jsonl style of full examples
            data_dir_list, # where images are saved
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()


    def get_data_paths(
        self, 
        json_dir_list, 
        data_dir_list, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_dir, image_dir in zip(
            json_dir_list, data_dir_list
        ):
            all_data = []
            for file in os.listdir(jsonl_dir):
                # read eveyrthing in so we can do global shuffle. 
                # TODO, we may preshuffe and shard.
                jsonl_path = os.path.join(jsonl_dir, file)
                with open(jsonl_path, 'r') as f:
                    raw_data = f.readlines()
                if shuffle_lines:
                    self.rng.seed(shuffle_seed)
                    self.rng.shuffle(raw_data)
                all_data += raw_data
            data_paths.extend([(json_data, image_dir) for json_data in all_data])
        print(f"image_dir: {image_dir}")
        print(f"Done loading for task {self.dataset_name} with {len(data_paths)} examples")
        return data_paths


    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(data)
                except:
                    traceback.print_exc()
                    continue
                try:
                    data = self._init_data()
                    # TODO: update the iamge path to be relative path
                    source_image_path = os.path.join(image_dir, data_item["source_image"])
                    target_image_path = os.path.join(image_dir, data_item["target_image"])
                    if not os.path.exists(source_image_path) or not os.path.exists(target_image_path):
                        print(f"source_image_path: {source_image_path} or target_image_path: {target_image_path} does not exist")
                        continue
                    condition_image = Image.open(source_image_path)
                    edited_image = Image.open(target_image_path)
                    edit_instruction = data_item["instruction"]
                    
                    data = self._add_image(
                        data, 
                        pil_img2rgb(condition_image),
                        need_loss=False, 
                        need_vae=True, 
                        need_vit=True, 
                    )
                    data = self._add_text(data, edit_instruction, need_loss=False)
                    data = self._add_image(
                        data, 
                        pil_img2rgb(edited_image),
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
                except Exception as e:
                    print(
                        f"Error when trying to decode line {row_idx} {e}"
                    )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
