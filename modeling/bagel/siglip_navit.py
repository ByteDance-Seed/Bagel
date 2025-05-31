# Copyright (c) 2024 The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.

import torch
from torch import nn

from transformers.activations import ACT2FN
from modeling.siglip.configuration_siglip import SiglipVisionConfig as _SiglipVisionConfig
from modeling.siglip.modeling_siglip import SiglipAttention, SiglipPreTrainedModel
from flash_attn import flash_attn_varlen_func


class SiglipVisionConfig(_SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`SiglipVisionModel`]. It is used to instantiate a
    Siglip vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import SiglipVisionConfig, SiglipVisionModel

    >>> # Initializing a SiglipVisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipVisionConfig()

    >>> # Initializing a SiglipVisionModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        rope=True,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs)
        
        self.rope = rope


class RotaryEmbedding2D(torch.nn.Module):
    def __init__(self, dim, max_h, max_w, base=10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base ** freq)

        grid_h = torch.arange(0, max_h)
        grid_h = grid_h.to(inv_freq.dtype)
        grid_h = grid_h[:, None].repeat(1, max_w)

        grid_w = torch.arange(0, max_w)
        grid_w = grid_w.to(inv_freq.dtype)
        grid_w = grid_w[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)

        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    def _forward_one_side(self, grid, inv_freq):
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # unsqueeze due to the head dimension
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        if not config.rope:
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def convert_conv2d_to_linear(self, config, meta=False):
        if meta:
            linear_patch_embedding = nn.Linear(
                config.num_channels * self.patch_size ** 2, self.embed_dim, bias=True, device='meta'
            )
        else:
            linear_patch_embedding = nn.Linear(
                config.num_channels * self.patch_size ** 2, self.embed_dim, bias=True
            )
        W = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
            self.embed_dim, config.num_channels * self.patch_size ** 2
        )
        linear_patch_embedding.weight.data = W
        linear_patch_embedding.bias.data = self.patch_embedding.bias.data
        del self.patch_embedding
        self.patch_embedding = linear_patch_embedding

    def forward(
        self, 
        packed_pixel_values: torch.FloatTensor, 
        packed_flattened_position_ids: torch.LongTensor
    ) -> torch.Tensor:

        patch_embeds = self.patch_embedding(packed_pixel_values)
        if not self.config.rope:
            embeddings = patch_embeds + self.position_embedding(packed_flattened_position_ids)
        else:
            embeddings = patch_embeds
        return embeddings


class SiglipFlashAttention2(SiglipAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        total_q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(total_q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = query_states[:, :, :self.head_dim // 2], query_states[:, :, self.head_dim // 2:] 
            kh, kw = key_states[:, :, :self.head_dim // 2], key_states[:, :, self.head_dim // 2:]
            qh, kh = apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        attn_output = flash_attn_varlen_func(
            query_states.to('cuda', torch.bfloat16),
            key_states.to('cuda', torch.bfloat16),
            value_states.to('cuda', torch.bfloat16),
            cu_seqlens_q=cu_seqlens.to('cuda'),
            cu_seqlens_k=cu_seqlens.to('cuda'),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )

        attn_output = self.out_proj(attn_output.reshape(total_q_len, -1))
        return attn_output


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipFlashAttention2(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cos_h=cos_h,
            sin_h=sin_h,
            cos_w=cos_w,
            sin_w=sin_w
        )
        hidden_states = residual + hidden_states.to(device=residual.device, dtype=residual.dtype)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states.to(device=residual.device, dtype=residual.dtype)

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
    ) -> torch.Tensor:

        hidden_states = inputs_embeds.to('cuda', torch.bfloat16)
        cu_seqlens = cu_seqlens.to('cuda', torch.int32)
        cos_h = cos_h.to('cuda', torch.bfloat16) if cos_h is not None else None
        sin_h = sin_h.to('cuda', torch.bfloat16) if sin_h is not None else None
        cos_w = cos_w.to('cuda', torch.bfloat16) if cos_w is not None else None
        sin_w = sin_w.to('cuda', torch.bfloat16) if sin_w is not None else None
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen,
                                          cos_h=cos_h, sin_h=sin_h, cos_w=cos_w, sin_w=sin_w)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = RotaryEmbedding2D(dim_head // 2, max_size, max_size)

        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            packed_pixel_values=packed_pixel_values, 
            packed_flattened_position_ids=packed_flattened_position_ids
        ).to('cuda')
        
        packed_pixel_values = packed_pixel_values.to('cuda')
        cu_seqlens = cu_seqlens.to('cuda')
        
        packed_flattened_position_ids = packed_flattened_position_ids.to('cuda')

        extra_inputs = {}
        if self.config.rope:
            extra_inputs.update(
                cos_h = self.rope.cos_h[packed_flattened_position_ids],
                sin_h = self.rope.sin_h[packed_flattened_position_ids],
                cos_w = self.rope.cos_w[packed_flattened_position_ids],
                sin_w = self.rope.sin_w[packed_flattened_position_ids]
            )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, 
            **extra_inputs
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "packed_pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:

        return self.vision_model(
            packed_pixel_values=packed_pixel_values.to('cuda'),
            packed_flattened_position_ids=packed_flattened_position_ids.to('cuda'),
            cu_seqlens=cu_seqlens.to('cuda'),
            max_seqlen=max_seqlen,
        )
