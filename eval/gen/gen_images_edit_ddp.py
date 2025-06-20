# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from safetensors.torch import load_file
import random
import numpy as np
import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from modeling.bagel.qwen2_navit import NaiveCache
import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO
from inferencer_ddp  import InterleaveInferencer
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
import time



def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


DATA_DIR = "/home/liliyu/workspace/monopi/monopi/experimental/liliyu/export_wm/"


def setup_model(model_path, 
                checkpoint_step: int=-1, 
                run_name: str=None, 
                model_mode: str="raw", 
                max_latent_size: int=64,
                rank: int=0,
                device: str="cuda:0",
                ):
    ################################################################    
    # Init model 
    ################################################################
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

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

    ################################################################
    # Load model 
    ################################################################
    if not "fake" in args.model_path:
        if args.checkpoint_step == "-1":
            weight_path = args.model_path
        else:
            weight_path = os.path.join("results", args.run_name, "checkpoints", args.checkpoint_step)
        if args.model_mode == "ema":
            model_state_dict_path = os.path.join(weight_path, "ema.safetensors")
        elif args.model_mode == "raw":
            model_state_dict_path = os.path.join(weight_path, "model.safetensors")
        else:
            raise ValueError(f"Invalid model mode: {args.model_mode}")
        print(" ====== Loading model state dict from {} ====== ".format(model_state_dict_path))

        model_state_dict = load_file(model_state_dict_path, device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
        if rank == 0:
            print(msg)
        del model_state_dict

    model = model.to(device).to(torch.bfloat16).eval()
    vae_model = vae_model.to(device).to(torch.bfloat16).eval()
    gen_model = model
    
    return model, tokenizer, vae_model, gen_model, new_token_ids


def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    # parser.add_argument("--metadata_file", type=str, default="/home/liliyu/workspace/monopi/monopi/experimental/liliyu/export_wm/arx_leftarm/image_0/prompts.jsonl", help="JSONL file containing lines of metadata for each prompt.")
    parser.add_argument("--task_name", type=str, default="arx_step100")
    parser.add_argument("--image_key", type=str, default="image_0")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT')
    parser.add_argument('--checkpoint_step', type=str, default='-1')
    parser.add_argument('--run_name', type=str, default='SEED_part23_run0')
    parser.add_argument('--model_mode', type=str, default='ema')
    parser.add_argument('--wandb_project_name', type=str, default='bagel-edit-eval')
    args = parser.parse_args()
    ################################    
    # Init env 
    ################################
    seed = 42
    set_seed(seed)

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    model, tokenizer, vae_model, gen_model, new_token_ids = setup_model(
        model_path=args.model_path, 
        checkpoint_step=args.checkpoint_step, 
        run_name=args.run_name, 
        model_mode=args.model_mode, 
        max_latent_size=args.max_latent_size,
        device=device,
    )

    ################################################################
    # Set up inference and arguments
    ################################################################
    # Image Transform Preparing
    # ImageTransform, take max_image_size, min_image_size, image_stride
    # Image Transform Preparing
    vae_transform = ImageTransform(args.resolution*2, args.resolution, 16)
    vit_transform = ImageTransform(980, 224, 14)

    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )

    inference_hyper=dict(
        cfg_text_scale=3.0, # 4.0,
        cfg_img_scale=1.5, # 2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=2.0, #3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(args.resolution, args.resolution),
    )


    ################################################################
    # Set up output directory
    ################################################################
    output_dir = os.path.join("results", args.run_name, "editing_eval", args.checkpoint_step)

    gen_suffix = (f"renorm{inference_hyper['cfg_renorm_min']}_"
                 f"text{inference_hyper['cfg_text_scale']}_"
                 f"img{inference_hyper['cfg_img_scale']}_"
                 f"shift{inference_hyper['timestep_shift']}_"
                 f"res{inference_hyper['image_shapes'][0]}")


    gen_suffix = f'{args.task_name}_{args.image_key}_{args.model_mode}_{gen_suffix}'
    output_dir = os.path.join(output_dir,gen_suffix)
    os.makedirs(output_dir, exist_ok=True)  
    if rank == 0:
        print(f" ======= Output images are saved in {output_dir}  =======")

    ################################################################
    # Load metadata
    ################################################################
    # metadata_file = os.path.join(task_data_dir[args.task_name], args.image_key, "prompts.jsonl")
    metadata_file = os.path.join(DATA_DIR, args.task_name, args.image_key, "prompts.jsonl")
    with open(metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    total_metadatas = len(metadatas)
    
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")
    
    ################################################################
    # Distributed Inference
    ################################################################
    start_time = time.time()
    outpath = output_dir
    for idx in range(start, end):
        metadata = metadatas[idx]
        outpath = os.path.join(output_dir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        prompt = metadata['instruction']
        source_image = Image.open(metadata['source_image']).resize(inference_hyper['image_shapes'])
        if "target_image" in metadata:  
            target_image = Image.open(metadata['target_image']).resize(inference_hyper['image_shapes'])
            
        else:
            target_image = None
        flag = False
        for idx in range(args.num_images):
            if not os.path.exists(os.path.join(outpath, f"{idx:05}.png")):
                flag = False
                break
        if flag:
            print(f"GPU {rank} skipping generation for prompt: {prompt}")
            continue
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

        image_list = []
        for i in range(args.num_images // args.batch_size):
            output_dict = inferencer(image=source_image, text=prompt, device=device, **inference_hyper)
            image_list.append(output_dict['image'])

        sample_count = 0
        for sample in image_list:
            sample_count += 1
            # Crop the generated image
            print(f"Sample shape: {sample.size}")
            sample = sample.crop(sample.getbbox())
            
            if sample.size[0] != inference_hyper['image_shapes'][0]:
                resize_sample = sample.resize(inference_hyper['image_shapes'])
            else:
                resize_sample = None
            
            # Save the comparison image
            img_path = os.path.join(outpath, f"{idx:05}.png")
            source_image.save(os.path.join(outpath, f"{idx:05}_source.png"))
            sample.save(os.path.join(outpath, f"{idx:05}_edited.png"))
            if target_image is not None:
                target_image.save(os.path.join(outpath, f"{idx:05}_target.png"))
            if resize_sample is not None:
                resize_sample.save(os.path.join(outpath, f"{idx:05}_resized.png"))
            
    end_time = time.time()
    total_time = end_time - start_time
    num_prompts = end - start
    
    print(f"\nGPU {rank} Summary:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per prompt: {total_time/num_prompts:.2f} seconds")
    print(f"Number of prompts processed: {num_prompts}")

    print(f"GPU {rank} has completed all tasks, time cost: {time.time() - start_time} seconds")
    dist.barrier()
    
    ################################################################
    # Log images to wandb
    ################################################################    
    # Initialize wandb for this process
    import wandb
    if rank == 0:
        run_name = f'{args.run_name}_{args.image_key}_{args.checkpoint_step}_{gen_suffix}'
        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config={
                "num_images": args.num_images,
                "batch_size": args.batch_size,
                "rank": rank
            }, 
            resume="allow"
        )

        columns=["id", "image", "instruction", "edited", "target"]
        test_table = wandb.Table(columns=columns)


        all_images =  os.listdir(output_dir)
        print(f"found images  {all_images}")
        for img_dir in sorted(os.listdir(output_dir)):
            img_dir_path = os.path.join(output_dir, img_dir)
            if os.path.isdir(img_dir_path):
                # Log images from this directory
                metadata_path = os.path.join(img_dir_path, "metadata.jsonl")
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    prompt = metadata["instruction"]
                source_image_path = os.path.join(img_dir_path, "00000_source.png")
                edited_image_path = os.path.join(img_dir_path, "00000_edited.png")
                target_image_path = os.path.join(img_dir_path, "00000_target.png")
                assert os.path.exists(source_image_path)
                assert os.path.exists(edited_image_path)
                assert os.path.exists(target_image_path)
                test_table.add_data(img_dir, wandb.Image(source_image_path), prompt, wandb.Image(edited_image_path), wandb.Image(target_image_path))
        wandb.log({"Editing results" : test_table}, step=int(args.checkpoint_step))


    # Finish the wandb run
    wandb.finish()
        