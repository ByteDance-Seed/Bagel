# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from safetensors.torch import load_file

import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

import copy
from PIL import Image
from modeling.bagel.qwen2_navit import NaiveCache


# 分布式初始化函数
def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


def apply_scale(width, height, scale):
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))
    
    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = _make_divisible(new_width, 16)
    new_height = _make_divisible(new_height, 16)
    return new_width, new_height


def editing_image_with_think(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None,
    max_length=2048, simple_think=False,
):
    # set output size
    if img_size is None:
        w, h = images[0].size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]
    
    # system prompt
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    # FIXME: acutally not very suitable for video input
    for image in images:
        # add VAE
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vae_transform, 
            new_token_ids=new_token_ids,
            #timestep=0.0,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)

        # add ViT
        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)
        
    ########## think
    tmp_past_key_values = copy.deepcopy(past_key_values)
    tmp_newlens = copy.deepcopy(newlens)
    tmp_new_rope = copy.deepcopy(new_rope)
    tmp_generation_input, tmp_newlens, tmp_new_rope = gen_model.prepare_prompts(
        curr_kvlens=tmp_newlens,
        curr_rope=tmp_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        tmp_past_key_values = gen_model.forward_cache_update_text(tmp_past_key_values, **tmp_generation_input)  
    
    tmp_generation_input = gen_model.prepare_start_tokens(tmp_newlens, tmp_new_rope, new_token_ids)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_text(
            past_key_values=tmp_past_key_values,
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
    print("="*30, "original think", "="*30)
    print(think_output) 
    if simple_think:
        think_output_list = think_output.split("</think>")
        if think_output_list[1] != "":
            think_output = think_output_list[1].strip()
        print("="*30, "processed think", "="*30)
        print(think_output) 
    ########## think
    
    ##########  cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    
    ##########  cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    
    # system prompt
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    # add think_output
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )
    
    ##########  origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    # add think_output
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
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
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h//16, w//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h//8, w//8)
    tmpimage = vae_model.decode(latent)
    tmpimage = ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)
    
    return tmpimage, think_output


def editing_image(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None,
):
    # set output size
    if img_size is None:
        w, h = images[0].size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")


    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens, new_rope = [0], [0]

    # FIXME: acutally not very suitable for video input
    for image in images:
        # add VAE
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vae_transform, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)

        # add ViT
        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)

    # cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    # cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )
    
    # origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
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
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h//16, w//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h//8, w//8)
    tmpimage = vae_model.decode(latent)
    tmpimage = ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)

    return tmpimage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using CausalFusion model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSON file containing lines of metadata for each prompt")
    parser.add_argument("--cfg_text_scale", type=float, default=4)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--think", action="store_true")
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    args = parser.parse_args()
    
    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0:
        print(msg)

    del model_state_dict
    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    cfg_text_scale = args.cfg_text_scale
    cfg_img_scale = args.cfg_img_scale
    cfg_interval = [0., 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    with open(args.metadata_file, "r") as f:
        tmpdatas = json.load(f)
    metadatas = []
    for k, v in tmpdatas.items():
        tmp = v
        tmp['path'] = tmp['id']
        tmp['id'] = k
        metadatas.append(tmp)
    total_metadatas = len(metadatas)
    
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")
    image_path = "eval/gen/imgedit/Benchmark/singleturn"

    for idx in range(start, end):
        metadata = metadatas[idx]
        images = []
        images.append(pil_img2rgb(Image.open(os.path.join(image_path, metadata['path']))))
        prompt = metadata['prompt']
        outpath = os.path.join(output_dir, f"{metadata['id']}.png")
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

        if os.path.exists(outpath):
            print(f"GPU {rank} skipping generation for prompt: {prompt}")
            continue
        
        if args.think:
            tmpimage, think_output = editing_image_with_think(
                images=images,
                prompt=prompt,
                cfg_text_scale=cfg_text_scale, 
                cfg_img_scale=cfg_img_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                max_length=2048, 
                simple_think=False, 
            )
            with open(outpath.replace(".png", ".txt"), "w") as f:
                f.write(think_output)
        else:
            tmpimage = editing_image(
                images=images,
                prompt=prompt,
                cfg_text_scale=cfg_text_scale, 
                cfg_img_scale=cfg_img_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
            )

        tmpimage = tmpimage.crop(tmpimage.getbbox())
        tmpimage.save(outpath)

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()
