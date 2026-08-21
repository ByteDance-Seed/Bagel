# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import ast
import random
import unittest
from pathlib import Path


def load_dataset_iterator():
    source_path = Path(__file__).with_name("dataset_base.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PackedDataset"
    )
    iterator = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__iter__"
    )
    isolated_module = ast.Module(body=[iterator], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {"random": random}
    exec(compile(isolated_module, source_path, "exec"), namespace)
    return namespace["__iter__"]


class PackedDatasetBufferTests(unittest.TestCase):
    def test_retries_buffered_sample_after_flushing_current_pack(self):
        dataset_class = type(
            "PackedDatasetStub", (), {"__iter__": load_dataset_iterator()}
        )
        dataset = dataset_class()
        dataset.grouped_weights = [1.0]
        dataset.is_mandatory = [False]
        dataset.dataset_iters = [
            iter(
                [
                    {
                        "id": "a",
                        "num_tokens": 6,
                        "sequence_plan": [],
                        "data_indexes": "a",
                    },
                    {
                        "id": "b",
                        "num_tokens": 6,
                        "sequence_plan": [],
                        "data_indexes": "b",
                    },
                    {
                        "id": "c",
                        "num_tokens": 3,
                        "sequence_plan": [],
                        "data_indexes": "c",
                    },
                    {
                        "id": "d",
                        "num_tokens": 6,
                        "sequence_plan": [],
                        "data_indexes": "d",
                    },
                ]
            )
        ]
        dataset.max_num_tokens_per_sample = 10
        dataset.max_num_tokens = 10
        dataset.expected_num_tokens = 9
        dataset.prefer_buffer_before = 10
        dataset.max_buffer_size = 1
        dataset.set_sequence_status = lambda: {"curr": 0, "ids": [], "sample_lens": []}

        def pack_sequence(sample, sequence_status):
            sequence_status["curr"] += sample["num_tokens"]
            sequence_status["ids"].append(sample["id"])
            sequence_status["sample_lens"].append(sample["num_tokens"])
            return sequence_status

        dataset.pack_sequence = pack_sequence
        dataset.to_tensor = lambda sequence_status: {
            "ids": tuple(sequence_status["ids"])
        }

        iterator = iter(dataset)
        self.assertEqual(next(iterator)["ids"], ("a",))
        self.assertEqual(next(iterator)["ids"], ("b", "c"))


if __name__ == "__main__":
    unittest.main()
