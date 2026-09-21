# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Run with: python -m unittest data.test_transforms -v"""

import unittest

import torch

from data.transforms import MaxLongEdgeMinShortEdgeResize


def _resized_hw(transform, width, height, img_num=1):
    out = transform(torch.zeros(3, height, width), img_num=img_num)
    return out.shape[-2:]


class TestMaxLongEdgeMinShortEdgeResize(unittest.TestCase):

    def test_max_pixels_budget_is_used_not_squared(self):
        """`max_pixels` is an area budget, so the clamp must scale each edge by
        sqrt(budget / area). Using the raw area ratio as a per-edge factor shrinks
        the area to (ratio ** 2) * area, i.e. only `ratio` of the budget."""
        transform = MaxLongEdgeMinShortEdgeResize(
            max_size=1024, min_size=512, stride=16, max_pixels=512 * 512
        )
        height, width = _resized_hw(transform, 3000, 3000)
        self.assertEqual(width, height)  # square in, square out
        self.assertGreaterEqual(width * height, 0.9 * 512 * 512)

    def test_max_pixels_budget_with_multiple_images(self):
        """vlm_sft group of data/configs/example.yaml: a video sample calls the
        transform with img_num = number of sampled frames, so every frame gets
        max_pixels / img_num."""
        transform = MaxLongEdgeMinShortEdgeResize(
            max_size=980, min_size=378, stride=14, max_pixels=2_007_040
        )
        img_num = 12  # frame_sampler_args.max_num_frames
        budget = 2_007_040 / img_num
        for width, height in [(1920, 1080), (1280, 720), (980, 551)]:
            with self.subTest(width=width, height=height):
                out_h, out_w = _resized_hw(transform, width, height, img_num=img_num)
                self.assertGreaterEqual(out_w * out_h, 0.9 * budget)
                self.assertLessEqual(out_w * out_h, 1.1 * budget)
                self.assertAlmostEqual(out_w / out_h, width / height, delta=0.15)

    def test_untriggered_budget_leaves_edges_alone(self):
        """When the image already fits the budget, the long edge is capped at
        max_size and nothing else changes."""
        transform = MaxLongEdgeMinShortEdgeResize(
            max_size=1024, min_size=512, stride=16, max_pixels=14 * 14 * 9 * 1024
        )
        self.assertEqual(tuple(_resized_hw(transform, 2048, 1024)), (512, 1024))
        self.assertEqual(tuple(_resized_hw(transform, 800, 600)), (608, 800))

    def test_edges_are_divisible_by_stride(self):
        transform = MaxLongEdgeMinShortEdgeResize(
            max_size=980, min_size=378, stride=14, max_pixels=2_007_040
        )
        out_h, out_w = _resized_hw(transform, 1920, 1080, img_num=8)
        self.assertEqual(out_w % 14, 0)
        self.assertEqual(out_h % 14, 0)


if __name__ == "__main__":
    unittest.main()
