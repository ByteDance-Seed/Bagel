import os
import torch
import numpy as np
import argparse
import random
from PIL import Image

# 移除 gradio
# import gradio as gr 

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
# 如果你在本地没有安装 accelerate，可能需要处理一下，但通常运行大模型是必须的

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
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

def get_args():
    parser = argparse.ArgumentParser(description="BAGEL Local Inference with 2 Images")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--ByteDance-Seed-BAGEL-7B-MoT", help="Path to the model")
    parser.add_argument("--mode", type=int, default=1, help="1: bf16, 2: nf4, 3: int8")
    
    # 输入输出相关参数
    parser.add_argument("--img1", type=str, required=True, help="Path to the first image")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second image")
    parser.add_argument("--prompt", type=str, required=True, help="Instruction text")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the result image")
    
    # 推理超参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--show_thinking", action="store_true", help="Enable thinking process output")
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = get_args()
    set_seed(args.seed)

    print(f"Loading model from: {args.model_path}")
    
    # --- 1. 模型初始化 (保持原逻辑不变) ---
    model_path = args.model_path 
    
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
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # --- 2. 设备分配与加载 (保持原逻辑不变) ---
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
        'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map: device_map[k] = first_device

    print(f"Mode: {args.mode} (1=bf16, 2=nf4, 3=int8)")
    if args.mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema_merged.safetensors") if os.path.exists(os.path.join(model_path, "ema_merged.safetensors")) else os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif args.mode == 2: # NF4
        bnb_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif args.mode == 3: # INT8
        bnb_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError

    # 初始化推理器
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    # --- 3. 执行双图 + 指令推理 ---
    
    # 3.1 加载图片
    print(f"Loading Image 1: {args.img1}")
    print(f"Loading Image 2: {args.img2}")
    
    try:
        image1 = Image.open(args.img1).convert("RGB") # 确保转为RGB
        image2 = Image.open(args.img2).convert("RGB")
        # 将图片转为 RGB 格式，data_utils.pil_img2rgb 会在 inferencer 内部再次检查，但这里先处理更安全
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    input_images = [image1, image2]
    
    # 3.2 准备参数
    # 这些参数参考了原代码中 edit_image 的默认值
    inference_hyper = dict(
        max_think_token_n=1024 if args.show_thinking else 1024,
        do_sample=args.show_thinking, # 如果开启thinking才sample，否则默认False
        text_temperature=0.3,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0.0, 1.0],  
        timestep_shift=3.0,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    print(f"Processing prompt: '{args.prompt}'...")

    # 3.3 运行推理
    # 关键修改：将 input_images 列表传递给 image 参数
    result = inferencer(
        image_list=input_images, 
        text=args.prompt, 
        think=args.show_thinking, 
        **inference_hyper
    )

    # --- 4. 保存结果 ---
    output_img = result["image"]
    text_output = result.get("text", "")

    if output_img:
        output_img.save(args.output)
        print(f"Success! Image saved to: {args.output}")
    else:
        print("Failed to generate image.")

if __name__ == "__main__":
    main()