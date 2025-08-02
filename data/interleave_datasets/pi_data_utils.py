# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
This script demonstrates how to use PyTorch's DataLoader with a our datasets.

# If you are using this in a separate repo, make sure to do the following steps:
#  export MONOPI_REPO=/home/liliyu/workspace/monopi
#  pip install --requirement $MONOPI_REPO/monopi/model/data/requirements.txt
#  pip install -e $MONOPI_REPO
"""
import datetime
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
import socket
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
    hostname = socket.gethostname()
    slurm_job_id = os.environ["SLURM_JOB_ID"]
    real_local_rank = int(os.environ["LOCAL_RANK"])
    logging_prefix = f"create_pi_dataset(hostname-{hostname}, rank-{local_rank}, local-{real_local_rank})"
    mixture_path = f"/tmp/{config.exp_name}_task_mixture_{slurm_job_id}.pkl"
    print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] mixture path: {mixture_path}, before barrier.")
    torch.distributed.barrier()
    print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] mixture path: {mixture_path}, after barrier.")
    # print(f"Real local rank {real_local_rank}, global rank {local_rank}")
    if real_local_rank == 0:
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] entering mixture generation.")
        mp.set_start_method('spawn', force=True)
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] setting start method to spawn.")
        pickled_bytes = cloudpickle.dumps((config.data, mixture_path))
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] pickled bytes.")
        process = mp.Process(target=create_task_mixture_worker, args=(pickled_bytes,))
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] process created.")
        process.start()
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] process started, before barrier.")
        torch.distributed.barrier() # sync before wait loop
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] process started, after barrier.")
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
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, async_op=False)
            # print(f"rank {local_rank}, local rank {real_local_rank}: after reduction: {tensor.item()}")
            if tensor.item() == world_size:
                break
            time.sleep(30)
            print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] waiting for all rank")
        process.join()
    else:
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] entering mixture generation, before barrier.")
        torch.distributed.barrier()
        print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] entering mixture generation, after barrier.")
        while True:
            if not os.path.exists(mixture_path):
                tensor = torch.zeros(1, dtype=torch.int64)
            else:
                tensor = torch.ones(1, dtype=torch.int64)
            tensor = tensor.to(torch.cuda.current_device())
            # print(f"other rank {local_rank}, local rank {real_local_rank}: before reduction: {tensor.item()}")
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, async_op=False)
            # print(f"other rank {local_rank}, local rank {real_local_rank}: after reduction: {tensor.item()}")
            if tensor.item() == world_size:
                break
            time.sleep(30)
            print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] wait for all rank")
    print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] wrapping up, entering barrier.")
    torch.distributed.barrier()
    print(f"{logging_prefix}[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] wrapping up, after barrier.")
    mp.set_start_method('fork', force=True)

    with open(mixture_path, "rb") as f:
        task_mixture = cloudpickle.load(f)
        print(f"{logging_prefix} loading task mixture from {mixture_path}")

    mixture = task_mixture[split]
    dt = mixture.get_dataset(num_epochs=num_epochs, shuffle=True, shard_info=ShardInfo(local_rank, world_size))
    return dt
