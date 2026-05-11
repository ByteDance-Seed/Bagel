# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
import os
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset, JsonStandardIterableDataset
from ..data_utils import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data


class UnifiedSIIterableDataset(InterleavedBaseIterableDataset, JsonStandardIterableDataset):
    def __init__(
        self,
        *args,
        enable_vit_condition=True,
        add_context_role_text=False,
        **kwargs,
    ):
        self.enable_vit_condition = enable_vit_condition
        self.add_context_role_text = add_context_role_text
        super().__init__(*args, **kwargs)

    @staticmethod
    def _context_role_text(idx):
        if idx == 0:
            return "This is the first image."
        if idx == 1:
            return "This is the second image."
        return f"This is context image {idx + 1}."

    def parse_row(self, row, image_dir):
        data = self._init_data()

        context_list = row.get("context", [])
        if not isinstance(context_list, list) or len(context_list) == 0:
            return {}

        for idx, rel_path in enumerate(context_list):
            full_path = os.path.join(image_dir, rel_path)
            try:
                img = pil_img2rgb(Image.open(full_path))
            except Exception as e:
                print(f"Error loading context image {rel_path}: {e}")
                return {}

            if self.add_context_role_text:
                data = self._add_text(
                    data,
                    self._context_role_text(idx),
                    need_loss=False,
                    enable_cfg=False,
                )

            data = self._add_image(
                data,
                img,
                need_loss=False,
                need_vae=True,
                need_vit=self.enable_vit_condition,
            )

        instruction = row.get("instruction", "")
        if not isinstance(instruction, str) or len(instruction.strip()) == 0:
            return {}
        data = self._add_text(data, instruction, need_loss=False)

        target_rel = row.get("target")
        if not isinstance(target_rel, str) or len(target_rel) == 0:
            return {}

        full_path = os.path.join(image_dir, target_rel)
        try:
            tgt_img = pil_img2rgb(Image.open(full_path))
            data = self._add_image(
                data,
                tgt_img,
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
        except Exception as e:
            print(f"Error loading target image {target_rel}: {e}")
            return {}

        return data
