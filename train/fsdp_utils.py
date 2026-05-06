# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch import nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    # FSDP v2 api below
    fully_shard,
    register_fsdp_forward_method,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
)
from torch.distributed.tensor import DeviceMesh, distribute_tensor 
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )

def fsdp_v2_shard(model, fsdp_config, cpu_offload=False):
    """
    Initializes and shards a model using FSDP v2 with a comprehensive bottom-up strategy.
    """
    # 1. Boilerplate: Setup device mesh and policies
    if fsdp_config.sharding_strategy == "HYBRID_SHARD":
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()))

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        # output_dtype=torch.bfloat16,
    )
    offload_policy = CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy()

    # 3. Explicit Bottom-Up Sharding: Wrap every module with parameters    
    for layer in model.language_model.model.layers:
        # Shard each MoT layer
        fully_shard(layer, mesh=device_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        # Register forward methods for fsdp v2 to recongnize dynamic functions that's not named "forward"
        register_fsdp_forward_method(layer, "forward_train")
        register_fsdp_forward_method(layer, "forward_inference")

    # (Optional) Shard the Bagel-specific visual generation modules
    fully_shard(model.time_embedder, mesh=device_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
    fully_shard(model.latent_pos_embed, mesh=device_mesh, mp_policy=mp_policy, offload_policy=offload_policy)

    fully_shard(
        model, 
        mesh=device_mesh, 
        mp_policy=mp_policy, 
        offload_policy=offload_policy, 
    )
    # Register forward methods for fsdp v2 to recongnize dynamic functions that's not named "forward"
    register_fsdp_forward_method(model, "forward_cache_update_text")
    register_fsdp_forward_method(model, "forward_cache_update_vit")
    register_fsdp_forward_method(model, "forward_cache_update_vae")
    register_fsdp_forward_method(model, "generate_image")
    register_fsdp_forward_method(model, "_forward_flow")

    return model

class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_ckpt_fsdp_v2(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        """The FSDP v2 version of DCP checkpoint loading that largely replicate existing try_load_ckpt()"""
        device = dist.get_rank() % torch.cuda.device_count()
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            # model_state_dict.pop('latent_pos_embed.pos_embed')
            # model_state_dict.pop('vit_pos_embed.pos_embed')

            meta_sd_main = model.state_dict()
            final_sd_main = {}
            # 2. Load all parameters that exist in the checkpoint
            for param_name, full_tensor in model_state_dict.items():
                if param_name not in meta_sd_main:
                    logger.warning(
                        f"[Main Model] Checkpoint key '{param_name}' not found in model definition. Skipping."
                    )
                    continue
                meta_param = meta_sd_main[param_name]
                if hasattr(meta_param, "device_mesh"):  # It's a sharded DTensor
                    sharded_tensor = distribute_tensor(
                        full_tensor.to(meta_param.dtype),
                        device_mesh=meta_param.device_mesh,
                        placements=meta_param.placements,
                    )
                else:  # It's a replicated tensor
                    sharded_tensor = full_tensor.to(
                        device=device, dtype=meta_param.dtype
                    )

                final_sd_main[param_name] = nn.Parameter(sharded_tensor)

            # msg = model.load_state_dict(model_state_dict, strict=False)
            # because we popped pos_embed
            msg = model.load_state_dict(final_sd_main, assign=True, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                # ema_state_dict.pop('latent_pos_embed.pos_embed')
                # ema_state_dict.pop('vit_pos_embed.pos_embed')

                meta_sharded_sd_ema = ema_model.state_dict()
                sharded_sd_ema = {}
                # 2. Load all parameters that exist in the checkpoint
                for param_name, full_tensor in ema_state_dict.items():
                    if param_name not in meta_sharded_sd_ema:
                        logger.warning(
                            f"[Main Model] Checkpoint key '{param_name}' not found in model definition. Skipping."
                        )
                        continue
                    meta_param = meta_sharded_sd_ema[param_name]
                    if hasattr(meta_param, "device_mesh"):  # It's a sharded DTensor
                        sharded_tensor = distribute_tensor(
                            full_tensor.to(meta_param.dtype),
                            device_mesh=meta_param.device_mesh,
                            placements=meta_param.placements,
                        )
                    else:  # It's a replicated tensor
                        sharded_tensor = full_tensor.to(
                            device=device, dtype=meta_param.dtype
                        )

                    sharded_sd_ema[param_name] = nn.Parameter(sharded_tensor)
                # because we popped pos_embed
                msg = ema_model.load_state_dict(sharded_sd_ema, assign=True, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)

@torch.no_grad()
def ema_step_fsdp_v2(model, ema_model, decay):
    """
    Update the exponential moving average (EMA) model's parameters.

    This function iterates through the parameters of the main model and EMA model,
    applying the EMA update rule in-place to the EMA model's parameters. It is
    designed to work with FSDP v2 by handling mixed DTensor and regular Tensor
    types, which would cause errors with torch._foreach operations.
    """
    main_params = model.parameters()
    ema_params = ema_model.parameters()

    for p_ema, p_main in zip(ema_params, main_params):
        # Only update the EMA for parameters that are being trained
        if p_main.requires_grad:
            # The EMA update rule: p_ema = decay * p_ema + (1 - decay) * p_main
            # This is performed in-place on the EMA parameter's data.
            p_ema.data.mul_(decay).add_(p_main.data, alpha=1 - decay)