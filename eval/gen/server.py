# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import torch
import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image
from safetensors.torch import load_file

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

import websockets
import logging
import msgpack_numpy
import gc
import functools as ft
import traceback
import argparse
import asyncio
from msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)
PRETRAINED_PATH = "pretrained_models/BAGEL-7B-MoT"
model_path = PRETRAINED_PATH

def prepare_model(weights_path: str):
    print(f"Loading model from {weights_path}") 
    # Model Initialization
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)


    ckpt_path = os.path.join(weights_path, "model.safetensors")
    print(f"Loading checkpoint from {ckpt_path}")
    

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        # model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Model Loading and Multi GPU Infernece Preparing
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "32GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        print("single gpu")
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    print(device_map)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ckpt_path,
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    print("===================  Model loaded successfully  ==============")

    # Inferencer Preparing 
    vae_transform = ImageTransform(224, 224, 16)
    vit_transform = ImageTransform(224, 224, 14)
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer

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

inference_hyper = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.2,
    cfg_interval=[0.0, 1.0],
    timestep_shift=2.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
    image_shapes=(224, 224),
)

packer = Packer()


#### SERVER SIDE ####
async def handler(websocket, inferencer):
    try:
        logger.info(f"Connection from {websocket.remote_address} opened")
        
        def infer(inputs: dict, cfg_text_scale=4.0, cfg_img_scale=1.5) -> dict:
            set_seed(42)
            source_image = Image.fromarray(inputs['observation/base_0_camera/rgb/image'])
            source_image.save("ur5s4_b1_source.png")
            prompt = inputs['raw_text']
            print(prompt)
            inference_hyper['cfg_text_scale'] = cfg_text_scale
            inference_hyper['cfg_img_scale'] = cfg_img_scale
            output_dict = inferencer(image=source_image, text=prompt, **inference_hyper)
            prompt = prompt.replace(" ", "_")
            output_dict['image'].save(f"ur5s4_b1_edited_{prompt}.png")
            outputs = {}
            outputs["future/observation/base_0_camera/rgb/image"] = np.array(output_dict['image'])
            return outputs

        logger.info("Model loaded.")
        while True:
            try:
                payload = await websocket.recv()
                args, kwargs = unpackb(payload)
                outputs = infer(*args, **kwargs)
                payload = packer.pack(outputs)
                await websocket.send(payload)
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
    except Exception as e:
        await websocket.send(traceback.format_exc())
        await websocket.close(
            code=websockets.frames.CloseCode.INTERNAL_ERROR,
            reason="Internal server error. Traceback included in previous frame.",
        )
        logger.error(f"Exception occurred: {e}")
        raise
    finally:
        # del inferencer
        gc.collect()


async def main(args):
    """
    Starts the WebSocket server.
    """
    inferencer = prepare_model(args.weights_path)

    print(f"Starting WebSocket server on ws://localhost:{args.port}")
    # async with websockets.serve(ft.partial(handler, ckpt_dir=args.weights_path), "0.0.0.0", args.port, compression=None, max_size=None) as server:
    #     await server.wait_closed()  # Run forever
    async with websockets.serve(ft.partial(handler, inferencer=inferencer), "0.0.0.0", args.port, compression=None, max_size=None) as server:
        await server.wait_closed()  # Run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--weights_path", type=str, default="/home/liliyu/workspace/BAGEL/results/pi_ur5e4_b1_endspan_lang_seedp1_gpu8_seq16384/checkpoints/0030000")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    asyncio.run(main(args))