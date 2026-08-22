import importlib
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import patch


def module(name, **attributes):
    value = types.ModuleType(name)
    value.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, attribute in attributes.items():
        setattr(value, key, attribute)
    return value


modeling = module("modeling")
modeling.__path__ = []
bagel = module("modeling.bagel")
bagel.__path__ = []

modeling_class = type("ModelingClass", (), {})
stubs = {
    "modeling": modeling,
    "modeling.bagel": bagel,
    "modeling.bagel.modeling_utils": module(
        "modeling.bagel.modeling_utils",
        MLPconnector=modeling_class,
        PositionEmbedding=modeling_class,
        TimestepEmbedder=modeling_class,
    ),
    "modeling.bagel.qwen2_navit": module(
        "modeling.bagel.qwen2_navit",
        Qwen2DecoderLayer=modeling_class,
        Qwen2MoEDecoderLayer=modeling_class,
        Qwen2MoTDecoderLayer=modeling_class,
    ),
    "modeling.bagel.siglip_navit": module(
        "modeling.bagel.siglip_navit",
        SiglipEncoderLayer=modeling_class,
        SiglipVisionTransformer=modeling_class,
    ),
}

with patch.dict(sys.modules, stubs):
    fsdp_utils = importlib.import_module("train.fsdp_utils")


class HybridMeshShapeTest(unittest.TestCase):
    def test_infers_replicas_for_multinode_world(self):
        self.assertEqual(fsdp_utils.resolve_hybrid_mesh_shape(None, 8, 16), (2, 8))

    def test_preserves_valid_explicit_shape(self):
        self.assertEqual(fsdp_utils.resolve_hybrid_mesh_shape(2, 8, 16), (2, 8))

    def test_rejects_explicit_shape_that_omits_ranks(self):
        with self.assertRaisesRegex(ValueError, r"1 \* 8 != 16"):
            fsdp_utils.resolve_hybrid_mesh_shape(1, 8, 16)

    def test_rejects_non_divisible_shard_count(self):
        with self.assertRaisesRegex(ValueError, "world_size to be divisible"):
            fsdp_utils.resolve_hybrid_mesh_shape(None, 6, 16)

    def test_wrapper_builds_mesh_with_all_distributed_ranks(self):
        config = fsdp_utils.FSDPConfig(
            sharding_strategy="HYBRID_SHARD",
            backward_prefetch="BACKWARD_PRE",
            cpu_offload=False,
            num_replicate=None,
            num_shard=8,
        )
        mesh = object()
        wrapped_model = object()

        with (
            patch.object(fsdp_utils.dist, "get_world_size", return_value=16),
            patch.object(fsdp_utils.dist, "get_rank", return_value=0),
            patch.object(fsdp_utils.torch.cuda, "device_count", return_value=8),
            patch.object(
                fsdp_utils, "init_device_mesh", return_value=mesh
            ) as init_mesh,
            patch.object(fsdp_utils, "FSDP", return_value=wrapped_model),
        ):
            result = fsdp_utils.fsdp_wrapper(object(), config)

        self.assertIs(result, wrapped_model)
        init_mesh.assert_called_once_with(
            "cuda",
            mesh_shape=(2, 8),
            mesh_dim_names=("replicate", "shard"),
        )


if __name__ == "__main__":
    unittest.main()
