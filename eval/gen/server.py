"""
PYTHONPATH=. python eval/gen/server.py  --weights_path /data/bagel_ckpts/pi_arx_biarm_allview_seq_seedp1_gpu16_seq16384/0050000/  --image_save_dir diverse_batch_folding   --image_key_str image_0,image_2,image_3
"""

import numpy as np
import os
import torch
import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image
from safetensors.torch import load_file
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
# from inferencer import InterleaveInferencer
from inferencer_video_ddp  import InterleaveInferencer
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

def prepare_model(weights_path: str, mode: int):
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
        visual_und=False,  # TODO: this can change due to the different model
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

    if args.mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=ckpt_path,
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif args.mode == 2: # NF4
        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
        model = load_and_quantize_model(
            model, 
            weights_location=ckpt_path, 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif args.mode == 3: # INT8
        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
        model = load_and_quantize_model(
            model, 
            weights_location=ckpt_path, 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError
    
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



packer = Packer()


image_key_map = {
    "image_0": "observation/base_0_camera/rgb/image",
    "image_1": "observation/base_1_camera/rgb/image",
    "image_2_ur5": "observation/wrist_0_camera/rgb/image",
    "image_2": "observation/left_wrist_0_camera/rgb/image",
    "image_3": "observation/right_wrist_0_camera/rgb/image",
}

#### SERVER SIDE ####
async def handler(websocket, inferencer, image_keys, image_dir, n_timesteps):
    try:
        logger.info(f"Connection from {websocket.remote_address} opened")
        inference_hyper = dict(
            cfg_text_scale=5.0,
            cfg_img_scale=1.2,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=n_timesteps,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=(224, 224),
        )
        def infer(inputs: dict, cfg_text_scale=4.0, cfg_img_scale=1.3) -> dict:
            set_seed(42)
            print(inputs.keys())
            file_count = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_images = [Image.fromarray(inputs[image_key_map[x]]) for x in image_keys]
            for i, source_image in enumerate(source_images):
                source_image.save(f"{image_dir}/{file_count}_source_{i}_{timestamp}.png")
            prompt = inputs['raw_text']
            print(prompt)
            inference_hyper['cfg_text_scale'] = cfg_text_scale
            inference_hyper['cfg_img_scale'] = cfg_img_scale
            image_list = inferencer.multiview_image_editing(source_images, prompt,  **inference_hyper)
            # output_dict = inferencer(image=image_list, text=prompt, **inference_hyper)
            prompt = prompt.replace(" ", "_")
            for i, image in enumerate(image_list):
                image.save(f"{image_dir}/edited_{file_count}_edited_{prompt}_{i}_{timestamp}.png")
            outputs = {}
            
            for image_key, image in zip(image_keys, image_list):
                outputs[f"future/{image_key_map[image_key]}"] = np.array(image)
            return outputs

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
    inferencer = prepare_model(args.weights_path, args.mode)
    image_keys = [x.strip() for x in args.image_key_str.split(",")]
    print(f"inferencing with image keys: {image_keys}")
    image_dir = f"generated_images/{args.image_save_dir}"
    os.makedirs(image_dir, exist_ok=True)
    print(f"generated images will be saved at: {image_dir}")

    print(f"\n\nStarting WebSocket server on ws://localhost:{args.port}")
    async with websockets.serve(ft.partial(handler, inferencer=inferencer, image_keys=image_keys, image_dir=image_dir, n_timesteps=args.n_timesteps), "0.0.0.0", args.port, compression=None, max_size=None) as server:
        await server.wait_closed()  # Run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--weights_path", type=str, default="/data/bagel_ckpts/pi_ur5e4_endspan_lange_seedp1_gpu8_seq16384/0040000/")
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--image_key_str", type=str, default="image_0,image_2_ur5")
    parser.add_argument("--image_save_dir", type=str, default="port_8000")
    parser.add_argument("--n_timesteps", type=int, default=25)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(main(args))