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
from monopi.model.configs import config as _config
from monopi.model.configs import registered_configs as register_cfg
from monopi.model.data import dataloader
from monopi.experimental.dibyaghosh import utils as experimental_utils
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
Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte



LOC_QUAD_RE = re.compile(
    #  whole match  v───────────────v  digits captured in groups 1-4
    r"(<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>)"
)

def convert_loc_to_bbox(text: str) -> str:
    """
    Replace every <loc####><loc####><loc####><loc####> sequence in *text*
    with a single <bbox> [...] </bbox> block.

    Ordering assumption in each quartet:
        <locYmin><locXmin><locYmax><locXmax>
    The output bbox order is:
        [xmin, ymin, xmax, ymax]
    """

    def _replace(match: re.Match) -> str:
        # Extract digit strings in the order ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = (int(g) / 1024 for g in match.groups()[1:])

        bbox_str = f"<bbox> [{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}] </bbox>"
        return bbox_str

    return LOC_QUAD_RE.sub(_replace, text)



MISTAKE_MAP = {
    0: False,
    1: True,
}  # -1: no mistake annotation

@dataclasses.dataclass(frozen=True)
class ShardInfo:
    """Information about a data source shard."""

    # Index of the current shard.
    shard_id: int = 0
    # Total number of shards.
    num_shards: int = 1

    
def create_pi_dataset_old(
    config: _config.TrainConfig, *, split: str = "train", num_epochs: int = 1, local_rank=0, world_size=1, rank0_only=False
):
    """Creates a PyTorch dataset from a config name."""
    print(f"rank-{local_rank} creating pi dataset naively")
    # create an dataset
    # experimental_utils.cache_specs(
    #     config.data.task_mixture_config, f"/home/{getpass.getuser()}/cached_specs"
    # )
    # config.data.max_episodes_per_task=10_000
    task_mixture = dataloader.create_task_mixture(config.data)
    mixture = task_mixture.mixtures[split]
    dt = mixture.get_dataset(num_epochs=num_epochs, shuffle=True, shard_info=ShardInfo(local_rank, world_size))
    return dt


def create_task_mixture_worker(pickled_bytes):
    config_data, output_path = cloudpickle.loads(pickled_bytes)
    task_mixture = dataloader.create_task_mixture(config_data).mixtures
    with open(output_path, "wb") as f:
        cloudpickle.dump(task_mixture, f)

def create_pi_dataset(
    config: _config.TrainConfig, *, split: str = "train", num_epochs: int = 1, local_rank=0, world_size=1, rank0_only=False
):
    """Creates a PyTorch dataset from a config name."""

    # mixture_path = f"/home/{getpass.getuser()}/{config.exp_name}_task_mixture.pkl"
    slurm_job_id = os.environ["SLURM_JOB_ID"]
    mixture_path = f"/tmp/{config.exp_name}_task_mixture_{slurm_job_id}.pkl"
    torch.distributed.barrier()
    real_local_rank = int(os.environ["LOCAL_RANK"])
    # print(f"Real local rank {real_local_rank}, global rank {local_rank}")
    if real_local_rank == 0:
        print(f"Real local rank {real_local_rank}, global rank {local_rank}, entering mixture generation.")
        mp.set_start_method('spawn', force=True)
        pickled_bytes = cloudpickle.dumps((config.data, mixture_path))
        process = mp.Process(target=create_task_mixture_worker, args=(pickled_bytes,))
        process.start()
        torch.distributed.barrier() # sync before wait loop
        # waiting loop, 
        # key idea is to contribute 1 when mixture_path exists and process finishes.
        # when all rank finishes, it will be equal to world size
        while True:
            if process.is_alive() or not os.path.exists(mixture_path):
                tensor = torch.zeros(1, dtype=torch.int64)
            else:
                tensor = torch.ones(1, dtype=torch.int64)
            tensor = tensor.to(torch.cuda.current_device())
            # print(f"rank {local_rank}, local rank {real_local_rank}: before reduction: {tensor.item()}")
            reduction_result = torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, async_op=False)
            # print(f"rank {local_rank}, local rank {real_local_rank}: after reduction: {tensor.item()}")
            if tensor.item() == world_size:
                break
            time.sleep(30)
            print(f"rank {local_rank}, local rank {real_local_rank}: Waiting for all rank")
        process.join()
    else:
        torch.distributed.barrier()
        while True:
            if not os.path.exists(mixture_path):
                tensor = torch.zeros(1, dtype=torch.int64)
            else:
                tensor = torch.ones(1, dtype=torch.int64)
            tensor = tensor.to(torch.cuda.current_device())
            # print(f"other rank {local_rank}, local rank {real_local_rank}: before reduction: {tensor.item()}")
            reduction_result = torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, async_op=False)
            # print(f"other rank {local_rank}, local rank {real_local_rank}: after reduction: {tensor.item()}")
            if tensor.item() == world_size:
                break
            time.sleep(30)
            print(f"other rank {local_rank}, local rank {real_local_rank}: wait for all rank")
    torch.distributed.barrier()
    mp.set_start_method('fork', force=True)

    with open(mixture_path, "rb") as f:
        task_mixture = cloudpickle.load(f)
        print(f"rank-{local_rank} loading task mixture from {mixture_path}")

    mixture = task_mixture[split]
    dt = mixture.get_dataset(num_epochs=num_epochs, shuffle=True, shard_info=ShardInfo(local_rank, world_size))
    return dt


class PiRobotQAAllViewsIterableDataset(InterleavedBaseIterableDataset):
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
                frames = []
                frame_indexes = []

                for image_key in self.image_key_list:
                    condition_image = row["image"][image_key]
                    condition_image = Image.fromarray(lib_image.decompress_image_if_needed(condition_image))
                    frames.append(condition_image)
                    frame_indexes.append(frame_idx)
                    frame_idx+= 1
                data = self._add_video(
                    data, 
                    frames,
                    frame_indexes,
                    need_loss=False, 
                    need_vae=False,
                    need_vit=True,
                )
                prompt = str(string_encode.decode_str(row["caption"]))
                try:
                    prompt = convert_loc_to_bbox(prompt)
                except Exception as e:
                    print(f"Error converting loc to bbox: {e}")
                    print(f"Prompt: {prompt}")
                    continue
                data = self._add_text(data, prompt, need_loss=True)

                
                if len(data) == 0:
                    continue
                # Add logger for debugging
                if row_idx <= self.n_log_examples:
                    # Create side-by-side full_example with text
                    self.save_example_multi_image(frames, frames, prompt, row_idx, self.image_key_list)
                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield data

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
