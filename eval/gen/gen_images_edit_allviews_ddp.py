# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
import random
import time
import shutil
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from PIL import Image

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer_video_ddp import InterleaveInferencer


def print_rank0(*args, **kwargs):
    """Print only on rank 0 process."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


DATA_DIR = "/home/liliyu/workspace/monopi/monopi/experimental/liliyu/export_wm/"


def setup_model(model_path, 
                checkpoint_step: int=-1, 
                run_name: Optional[str]=None, 
                model_mode: str="raw", 
                max_latent_size: int=64,
                rank: int=0,
                device: str="cuda:0",
                ):
    ################################################################
    # try find the weight path
    ################################################################
    if "fake" not in args.model_path:
        if args.checkpoint_step == "-1":
            weight_path = args.model_path
        else:
            weight_path = os.path.join(args.checkpoint_directory, args.run_name, "checkpoints", args.checkpoint_step)
        if args.model_mode == "ema":
            model_state_dict_path = os.path.join(weight_path, "ema.safetensors")
        elif args.model_mode == "raw":
            model_state_dict_path = os.path.join(weight_path, "model.safetensors")
        else:
            raise ValueError(f"Invalid model mode: {args.model_mode}")
        if not os.path.exists(weight_path):
            print_rank0(f"\n======= SKIPPING: Weight path does not exist: {weight_path} =======")
            exit()


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
    if "fake" not in args.model_path:
        print_rank0(f" ====== Loading model state dict from {model_state_dict_path} ====== ")
        # Load model state dict directly to target device to avoid CPU->GPU transfer
        model_state_dict = load_file(model_state_dict_path, device=str(device))
        msg = model.load_state_dict(model_state_dict, strict=False)
        print_rank0(msg)
        del model_state_dict
    
    print_rank0(" ====== Done loading, move model to devices ====== ")
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
    parser.add_argument("--image_key", type=str, default="all_views")
    parser.add_argument("--image_list_str", type=str, default="image_0,image_2")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT')
    parser.add_argument('--checkpoint_step', type=str, default='-1')
    parser.add_argument('--run_name', type=str, default='SEED_part23_run0')
    parser.add_argument('--model_mode', type=str, default='ema')
    parser.add_argument("--with_condition", type=bool, default=False)
    parser.add_argument('--wandb_project_name', type=str, default='bagel-edit-eval')
    parser.add_argument('--ids_for_fixed_prompt', type=int, default=-1)  #this is for fixed prompt, if -1, then use the prompt in the metadata
    parser.add_argument('--cfg_text_scale', type=float, default=5.0)
    parser.add_argument('--cfg_img_scale', type=float, default=1.2)
    parser.add_argument('--num_timesteps', type=int, default=25)
    parser.add_argument('--condition_on_vit', type=bool, default=False)
    parser.add_argument('--checkpoint_directory', type=str, default="/mnt/weka/checkpoints/liliyu/bagel_ckpt")
    args = parser.parse_args()

    seed = 42
    set_seed(seed)
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    editing_image_generated = False
    
    print_rank0(args)

    ################################################################
    # Set up output directory
    ################################################################
    inference_hyper=dict(
        cfg_text_scale=args.cfg_text_scale, # 4.0,
        cfg_img_scale=args.cfg_img_scale, # 2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0, #3.0,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        image_shapes=(args.resolution, args.resolution),
        using_2nd_order=False,
        with_condition=args.with_condition,
    )

    output_dir = os.path.join("/mnt/weka/checkpoints/liliyu/bagel_ckpt", args.run_name, "editing_eval", args.checkpoint_step)

    gen_suffix = (f"renorm{inference_hyper['cfg_renorm_min']}_"
                 f"text{inference_hyper['cfg_text_scale']}_"
                 f"img{inference_hyper['cfg_img_scale']}_"
                 f"shift{inference_hyper['timestep_shift']}_"
                 f"nsteps{inference_hyper['num_timesteps']}_"
                 f"res{inference_hyper['image_shapes'][0]}_"
                 f"heun{inference_hyper['using_2nd_order']}_"
                 f"condition{inference_hyper['with_condition']}_"
                 )

    gen_suffix = f'{args.task_name}_{args.image_key}_{args.model_mode}_{gen_suffix}'
    output_dir = os.path.join(output_dir,gen_suffix)
    os.makedirs(output_dir, exist_ok=True)  
    print_rank0(f"\n======= Output images are saved in {output_dir}  =======")

    # Check if editing images have already been generated
    subfolder_00019 = os.path.join(output_dir, "00019")
    if os.path.exists(subfolder_00019):
        png_files = [f for f in os.listdir(subfolder_00019) if f.endswith('.png')]
        if png_files:
            editing_image_generated = True
            print_rank0(f" \n======= Editing images have already been generated in {output_dir}  =======")
            exit()
            
    os.makedirs(output_dir, exist_ok=True)  

        
    ################################    
    # Init env 
    ################################

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
    print_rank0(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")
    
    ################################################################
    # Distributed Inference
    ################################################################
    image_keys = [k.strip() for k in args.image_list_str.split(",")] 
    print_rank0(f"GPU {rank}: image_keys: {image_keys}")
    N_images = len(image_keys)
    start_time = time.time()
    outpath = output_dir
    for idx in range(start, end):
        metadata = metadatas[idx]
        outpath = os.path.join(output_dir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        prompt = metadata['instruction'].split(";")[1].strip() 
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")
        
        source_images = []
        source_image_paths = []
        target_image_paths = []

        for img_path in metadata['source_images']:
            if sum(img_key in img_path for img_key in image_keys) == 1:
                source_image_paths.append(img_path)
                source_images.append(Image.open(img_path).resize(inference_hyper['image_shapes']))
                target_image_path = img_path.replace("_source_", "_target_")
                if target_image_path in metadata['target_images']:
                    target_image_paths.append(target_image_path)
                else:
                    target_image_paths.append(None)
        image_list = inferencer.multiview_image_editing(source_images, prompt, device=device, condition_on_vit=args.condition_on_vit, **inference_hyper )

        sample_count = 0
        for i, (sample, source_image, target_image) in enumerate(zip(image_list, source_image_paths, target_image_paths)):
            
            # Save the comparison image
            edited_image_path = os.path.join(outpath, f"view{i}_edited.png")
            if isinstance(sample, Image.Image):
                sample.save(edited_image_path)
            else:
                # If sample is a string/path, copy it
                shutil.copy(sample, edited_image_path)
            shutil.copy(target_image, os.path.join(outpath, f"view{i}_target.png"))
            shutil.copy(source_image, os.path.join(outpath, f"view{i}_source.png"))

    
    end_time = time.time()
    total_time = end_time - start_time
    num_prompts = end - start
    print_rank0(f"\nGPU {rank} Summary:")
    print_rank0(f"Total time: {total_time:.2f} seconds")
    print_rank0(f"Average time per prompt: {total_time/num_prompts:.2f} seconds")
    print_rank0(f"Number of prompts processed: {num_prompts}")
    print_rank0(f"GPU {rank} has completed all tasks, time cost: {time.time() - start_time} seconds")
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

        columns=["id", *[f"view{i}" for i in range(N_images)], "instruction",  *[f"edited_{i}" for i in range(N_images)], *[f"target_{i}" for i in range(N_images)]]
        test_table = wandb.Table(columns=columns)


        all_images =  os.listdir(output_dir)
        print_rank0(f"found images {all_images}")
        for img_dir in sorted(all_images):
            img_dir_path = os.path.join(output_dir, img_dir)
            if os.path.isdir(img_dir_path):
                # Log images from this directory
                metadata_path = os.path.join(img_dir_path, "metadata.jsonl")
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    prompt = metadata["instruction"]
                
                s_images = [wandb.Image(os.path.join(img_dir_path, f"view{i}_source.png")) for i in range(N_images)]
                e_images = [wandb.Image(os.path.join(img_dir_path, f"view{i}_edited.png")) for i in range(N_images)]
                t_images = [wandb.Image(os.path.join(img_dir_path, f"view{i}_target.png")) for i in range(N_images)]
                test_table.add_data(img_dir, *s_images, prompt, *e_images, *t_images)
        wandb.log({"Editing results" : test_table}, step=int(args.checkpoint_step))


    # Finish the wandb run
    wandb.finish()
        