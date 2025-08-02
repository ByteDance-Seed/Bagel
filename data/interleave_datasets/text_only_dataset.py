# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import traceback
import io
import random
from PIL import Image, ImageFile, PngImagePlugin
from pathlib import Path
from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
import wandb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class TextIterableDataset(InterleavedBaseIterableDataset):
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
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()


    def get_data_paths(self, data_dir_list, num_used_data):
        input_files = []
        for input_dir in data_dir_list:
            input_dir = Path(input_dir)
            for jsonl_file in sorted(input_dir.glob("*.jsonl")):
                input_files.append(str(jsonl_file))
        if self.local_rank == 0:
            print(f"for task {self.dataset_name}, gather all input_files: {input_files}")
        return input_files

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            json_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            json_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{json_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[json_start_id:] #skip the tar file that's alrady trained
            for jsonl_file_idx, jsonl_file_path in enumerate(data_paths_per_worker_, start=json_start_id):
                json_data = []
                with open(jsonl_file_path, 'r') as file:
                    for line in file:
                        json_data.append(json.loads(line.strip()))

                json_data = json_data[row_start_id:]
                for row_idx, row in enumerate(json_data, start=row_start_id):
                    # skip the row in this tar file that's already trained
                    try:
                        data = self._init_data()
                        
                        prompt = ""
                        for m in row['messages']:
                            prompt += m['role'] + ": " + m['content'] + "\n"
                        
                        # cap the prompt to 1024 tokens
                        # n_tokens = len(self.tokenizer.encode(prompt))
                        # if n_tokens > 15_000:
                        #     prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[:15_000])
                        #     print(f"prompt is too long, cap it to new prompt: {prompt}")

                        data = self._add_text(data, prompt, need_loss=True)
                        # Add a padded image to make sure a batch got an image. 
                        white_image = Image.new("RGB", (224, 224), (255, 255, 255))
                        image = pil_img2rgb(white_image)
                        data = self._add_image(
                            data,
                            image,
                            need_loss=False,
                            need_vae=False,
                            need_vit=True,
                        )
                        if row_idx <= self.n_log_examples:
                            # Create side-by-side full_example with text
                            self.save_example_image(image, image, prompt, row_idx)
                        data['data_indexes'] = {
                            "data_indexes": [jsonl_file_idx, row_idx],
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                        yield data
                    except Exception as e:
                        print(
                            f"Error when trying to decode line {row_idx} in {jsonl_file_idx} {e}"
                        )

                    row_start_id = 0
            json_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

