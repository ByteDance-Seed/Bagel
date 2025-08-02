# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pyarrow.parquet as pq
import random
from PIL import Image
from pathlib import Path
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
import webdataset as wds
from .interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2IIWebDataset(InterleavedBaseIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None, experiment_name=None, shuffle_seed=0,n_log_examples=100,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.experiment_name = experiment_name
        self.shuffle_seed = shuffle_seed
        self.n_log_examples = n_log_examples
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        input_files = []
        for input_dir in data_dir_list:
            input_dir = Path(input_dir)
            for tarfile in sorted(input_dir.glob("*.tar")):
                input_files.append(str(tarfile))
        return input_files


    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            tar_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            tar_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{tar_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[tar_start_id:] #skip the tar file that's alrady trained
            for tarfile_idx, tar_file_path in enumerate(data_paths_per_worker_, start=tar_start_id):
                    wds_obj = wds.WebDataset(tar_file_path, nodesplitter=lambda x: x).shuffle(10000)

                    try:
                        peek = next(iter(wds_obj))
                    except StopIteration:
                        print(f"[Warning] {tar_file_path} is empty. Skipping.")
                        continue

                    for row_idx, row in enumerate(wds.WebDataset(tar_file_path, nodesplitter=lambda x: x).shuffle(10000)):
                        # skip the row in this tar file that's already trained
                        if row_idx < row_start_id:
                            continue
                        try:
                            data = self._init_data()
                            
                            image_byte = row['jpg']
                            image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            caption = row['txt'].decode('utf-8') if isinstance(row['txt'], bytes) else row['txt']
                            data = self._add_text(data, f"Generate image from caption: {caption}", need_loss=False)
                            data = self._add_image(
                                data,
                                image,
                                need_loss=True,
                                need_vae=False,
                                need_vit=False,
                                enable_cfg=False,
                            )
                            if row_idx <= self.n_log_examples:
                                # Create side-by-side full_example with text
                                self.save_example_image(image, image, caption, row_idx)
                            data['data_indexes'] = {
                                "data_indexes": [tarfile_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                            yield data
                        except Exception as e:
                            print(
                                f"Error when trying to decode line {row_idx} in {tar_file_path} {e}"
                            )

                    row_start_id = 0
            tar_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")




class I2TIWebDataset(InterleavedBaseIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None, experiment_name=None, shuffle_seed=0,n_log_examples=100,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.experiment_name = experiment_name
        self.shuffle_seed = shuffle_seed
        self.n_log_examples = n_log_examples
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        input_files = []
        for input_dir in data_dir_list:
            input_dir = Path(input_dir)
            for tarfile in sorted(input_dir.glob("*.tar")):
                input_files.append(str(tarfile))
        return input_files


    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            tar_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            tar_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{tar_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[tar_start_id:] #skip the tar file that's alrady trained
            for tarfile_idx, tar_file_path in enumerate(data_paths_per_worker_, start=tar_start_id):
                    wds_obj = wds.WebDataset(tar_file_path, nodesplitter=lambda x: x).shuffle(10000)

                    try:
                        peek = next(iter(wds_obj))
                    except StopIteration:
                        print(f"[Warning] {tar_file_path} is empty. Skipping.")
                        continue

                    for row_idx, row in enumerate(wds.WebDataset(tar_file_path, nodesplitter=lambda x: x).shuffle(10000)):
                        # skip the row in this tar file that's already trained
                        if row_idx < row_start_id:
                            continue
                        try:
                            data = self._init_data()
                            
                            image_byte = row['jpg']
                            image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            caption = row['txt'].decode('utf-8') if isinstance(row['txt'], bytes) else row['txt']
                            data = self._add_image(
                                data,
                                image,
                                need_loss=False,
                                need_vae=False,
                                need_vit=True,
                                enable_cfg=False,
                            )
                            data = self._add_text(data, f"Generate image from caption: {caption}", need_loss=True)
                            if row_idx <= self.n_log_examples:
                                # Create side-by-side full_example with text
                                self.save_example_image(image, image, caption, row_idx)
                            data['data_indexes'] = {
                                "data_indexes": [tarfile_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                            yield data
                        except Exception as e:
                            print(
                                f"Error when trying to decode line {row_idx} in {tar_file_path} {e}"
                            )

                    row_start_id = 0
            tar_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")



class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['image']
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                caption_dict = row['captions']
                                caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                            if len(caps_token) == 0:
                                print(f'no caption in rg#{row_group_id}, {parquet_file_path}')
                                caption_token = self.tokenizer.encode(' ')
                            else:
                                caption_token = random.choice(caps_token)

                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                        
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
