# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
import copy
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from io import BytesIO
from safetensors.torch import load_file
from pathlib import Path
from datasets import load_dataset

# Import model components
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, 
    Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from data.transforms import ImageTransform
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_models(model_path, device=0):
    """
    Set up and load all required models.
    
    Args:
        llm_path: Path to the language model
        vit_path: Path to the vision model
        vae_path: Path to the VAE model
        device: GPU device ID
        
    Returns:
        tuple: (model, vae_model, tokenizer, new_token_ids, vae_transform)
    """
    # Set device
    torch.cuda.set_device(device)
    
    # Load LLM
    llm_config = Qwen2Config.from_pretrained(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # Load Vision model
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # Load VAE
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # Create fusion model
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    
    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    ema_state_dict_path = os.path.join(model_path, f"ema.safetensors") # may beed to change
    ema_state_dict = load_file(ema_state_dict_path, device="cpu")
    msg = model.load_state_dict(ema_state_dict, strict=False)


    # Set up transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)
    
    
    # Move models to GPU
    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    
    return model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform


def editing_with_text_img_cfg(
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
    image, prompt,
    num_timesteps=50,
    cfg_text_scale=8.0,
    cfg_img_scale=1.5,
    cfg_type="parallel",
    cfg_interval=[0.4, 1.0],
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
    timestep_shift=3.0,
    temperature=0.3,
    # Image transform params
    max_image_size=512,
    min_image_size=256,
    stride=16,
    seed=42,
    use_vit=False,
    use_think=False,
    # output_size=(576, 1024)  # Default size, can be changed
    device='cuda'
):
    """
    Edit an image based on text instructions using NAVIT model.
    
    Args:
        model: The CausalFusion model
        vae_model: The VAE model
        tokenizer: Tokenizer for text processing
        new_token_ids: Special token IDs
        vae_transform: Transform for VAE input
        image: Input PIL image
        prompt: Text prompt for editing
        num_timesteps: Number of diffusion steps
        cfg_text_scale: Text CFG scale
        cfg_img_scale: Image CFG scale
        cfg_type: CFG type (parallel or serial)
        cfg_interval: CFG interval
        cfg_renorm_min: CFG renormalization minimum value
        cfg_renorm_type: CFG renormalization type
        timestep_shift: Timestep shift for diffusion
        max_image_size: Maximum size for image dimension
        min_image_size: Minimum size for image dimension
        stride: Stride for resizing
        seed: Random seed
        output_size: Tuple of (height, width) for output image
        
    Returns:
        PIL.Image: Edited image
    """
    
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = _make_divisible(new_width, stride)
        new_height = _make_divisible(new_height, stride)
        return new_width, new_height
    
    
    # Set random seeds for reproducibility
    set_seeds(seed)
    
    # Prepare image size
    w, h = image.size
    scale = min(max_image_size / max(w, h), 1.0)
    scale = max(scale, min_image_size / min(w, h))
    w, h = _apply_scale(w, h, scale)
    
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = _apply_scale(w, h, scale)
    
    # print(f"Image size: H-{h} W-{w}")

    # Initialize cache and setup
    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]
    
    if use_think:
        SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''
        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[SYSTEM_PROMPT],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = move_generation_input_to_device(generation_input, device)
            past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)  
        
    
    # Prepare VAE images
    generation_input, newlens, new_rope = model.prepare_vae_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=[image],
        transforms=vae_transform, 
        new_token_ids=new_token_ids,
        timestep=0.0,
    )

    # Forward pass for VAE cache update
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)
            
    if use_vit:
        generation_input, newlens, new_rope = model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = move_generation_input_to_device(generation_input, device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

    # Setup for text CFG
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    
    # Prepare prompts for main branch
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )

    # Forward pass for main branch
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)  
    
    
    if use_think: 
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            tmp_generation_input = model.prepare_start_tokens(newlens, new_rope, new_token_ids)
            
            tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
            unpacked_latent = model.generate_text(
                past_key_values=copy.deepcopy(past_key_values),
                max_length=10240,
                do_sample=True,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **tmp_generation_input,
                )
            output = tokenizer.decode(unpacked_latent[:,0])
            think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1] 
            
        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[think_output],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = move_generation_input_to_device(generation_input, device)
            past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)  
        
    # Prepare VAE latent for main branch
    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )        

    # Setup for image CFG
    cfg_img_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    
    if use_think:
        cfg_img_texts = [SYSTEM_PROMPT, prompt, think_output]
    else:
        cfg_img_texts = [prompt]
    
    for text in cfg_img_texts:
        generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope, 
            prompts=[text],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )

        # Forward pass for image CFG
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
            cfg_img_past_key_values = model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)

    generation_input_cfg_img = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )


    # Extract packed positions and indexes for CFGs
    cfg_text_args = {
        'cfg_text_packed_position_ids': generation_input_cfg_text['cfg_packed_position_ids'],
        'cfg_text_packed_query_indexes': generation_input_cfg_text['cfg_packed_query_indexes'],
        'cfg_text_key_values_lens': generation_input_cfg_text['cfg_key_values_lens'],
        'cfg_text_packed_key_value_indexes': generation_input_cfg_text['cfg_packed_key_value_indexes'],
    }
    
    cfg_img_args = {
        'cfg_img_packed_position_ids': generation_input_cfg_img['cfg_packed_position_ids'],
        'cfg_img_packed_query_indexes': generation_input_cfg_img['cfg_packed_query_indexes'],
        'cfg_img_key_values_lens': generation_input_cfg_img['cfg_key_values_lens'],
        'cfg_img_packed_key_value_indexes': generation_input_cfg_img['cfg_packed_key_value_indexes'],
    }
    
    # Generate final image with mixed CFG
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        cfg_text_args = move_generation_input_to_device(cfg_text_args, device)
        cfg_img_args = move_generation_input_to_device(cfg_img_args, device)
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            **cfg_text_args,
            **cfg_img_args,
        )

    # Process and decode the latent representation
    latent0 = unpacked_latent[0]
    latent0 = latent0.reshape(1, h // 16, w // 16, 2, 2, 16)
    latent0 = torch.einsum("nhwpqc->nchpwq", latent0)
    latent0 = latent0.reshape(1, 16, h // 8, w // 8)
    image = vae_model.decode(latent0)
    
    # Convert to PIL image
    edited_image = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    edited_image = Image.fromarray(edited_image)

    return edited_image


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_dataset(
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
    output_dir, cfg_text_scale=4.0, cfg_img_scale=1.5, 
    cfg_type="serial_text_img", num_samples=None, shard_id=0, total_shards=1, use_think=False, device='cuda'
):
    """
    Process images from the dataset using the editing model.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("stepfun-ai/GEdit-Bench")['train']
    idx_list = list(range(len(dataset)))
    idx_list = idx_list[shard_id::total_shards]
    
    for data_idx in tqdm(idx_list):
        data = dataset[data_idx]
        
        task_type = data['task_type']
        key = data['key']
        instruction_language = data['instruction_language']

        save_path_fullset_source_image = f"{output_dir}/fullset/{task_type}/{instruction_language}/{key}_SRCIMG.png"
        save_path_fullset = f"{output_dir}/fullset/{task_type}/{instruction_language}/{key}.png"
        os.makedirs(os.path.dirname(save_path_fullset_source_image), exist_ok=True)
        os.makedirs(os.path.dirname(save_path_fullset), exist_ok=True)

        if os.path.exists(save_path_fullset_source_image) and os.path.exists(save_path_fullset):
            print(f'sample {key} already generated, skipping...')
            continue

        instruction = data["instruction"]
        input_image = data["input_image"]

        try:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                edited_image = editing_with_text_img_cfg(
                    model=model,
                    vae_model=vae_model,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    vae_transform=vae_transform,
                    vit_transform=vit_transform,
                    image=input_image,
                    prompt=instruction,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_type=cfg_type,
                    cfg_interval=[0.4, 1.0],
                    cfg_renorm_min=0.0,
                    cfg_renorm_type="text_channel",
                    timestep_shift=3.0,
                    seed=42,
                    max_image_size=vae_transform.resize_transform.max_size,
                    min_image_size=vae_transform.resize_transform.min_size,
                    use_vit=True,
                    use_think=use_think,
                    device=device,
                )

            input_image.save(save_path_fullset_source_image)
            edited_image.save(save_path_fullset)

        except Exception as e:
            raise
            print(f"Error processing image {key}: {e}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Image editing with text instructions")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to bagel model")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save output images")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0, help="Text CFG scale")
    parser.add_argument("--cfg_img_scale", type=float, default=1.5, help="Image CFG scale")
    parser.add_argument("--cfg_type", type=str, default="serial_text_img", 
                        help="CFG type (parallel, serial_text_img, etc.)")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to process (None for all)")
    parser.add_argument("--shard_id", type=int, default=0, 
                        help="ID of the current shard (0-based)")
    parser.add_argument("--total_shards", type=int, default=1, 
                        help="Total number of shards")
    parser.add_argument("--use_think", action='store_true', 
                        help="Whether enable thinking")
    
    args = parser.parse_args()
    
    # Setup models
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = setup_models(
        args.model_path, args.device
    )
    
    # Process dataset
    process_dataset(
        model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
        args.output_dir, 
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_type=args.cfg_type,
        num_samples=args.num_samples,
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        use_think=args.use_think,
        device=args.device,
    )


if __name__ == "__main__":
    main()