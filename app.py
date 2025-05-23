import gradio as gr
import numpy as np
import os
import torch

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

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


# Model Initialization
model_path = "/path/to/BAGEL-7B-MoT/weights" #Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

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

# Model Loading and Multi GPU Infernece Preparing
device_map = infer_auto_device_map(
    model,
    max_memory={i: "40GiB" for i in range(torch.cuda.device_count())},
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
            
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
).eval()

# Inferencer Preparing 
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

# t2i Inference Hyperparameters
t2i_inference_hyper = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="global",
)

# i2t Inference Hyperparameters
i2t_inference_hyper=dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
)

# t2i_think Inference Hyperparameters
t2i_think_inference_hyper=dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="global",
)

# i2i Inference Hyperparameters
i2i_inference_hyper=dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="text_channel",
)

# Gradio Interface Functions
def text_to_image(prompt):
    result = inferencer(text=prompt, **t2i_inference_hyper)
    return result["image"]

def image_understanding(image: Image.Image, prompt: str):
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    result = inferencer(image=image, text=prompt, understanding_output=True, **i2t_inference_hyper)
    return result["text"]

def reasoning_text_to_image(prompt):
    result = inferencer(text=prompt, think=True, **t2i_think_inference_hyper)
    
    return result["image"], result["text"]

def edit_image(image: Image.Image, prompt: str):
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    result = inferencer(image=image, text=prompt, **i2i_inference_hyper)
    return result["image"]

# Gradio UI 
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 BAGEL Playground")

    with gr.Tab("📝 text to image"):
        txt_input = gr.Textbox(label="Prompt")
        img_output = gr.Image(label="Image Generated")
        gen_btn = gr.Button("Submit")
        gen_btn.click(fn=text_to_image, inputs=txt_input, outputs=img_output)

    with gr.Tab("🖼️ image understanding"):
        img_input = gr.Image(label="Input Image")
        txt_input = gr.Textbox(label="Prompt")
        txt_output = gr.Textbox(label="Result")
        img_understand_btn = gr.Button("Submit")
        img_understand_btn.click(fn=image_understanding, inputs=[img_input, txt_input], outputs=txt_output)

    with gr.Tab("🧠 text to image with think"):
        reasoning_txt_input = gr.Textbox(label="Prompt")
        reasoning_text_output = gr.Textbox(label="Thinking Result")
        reasoning_img_output = gr.Image(label="Image Generated")
        reasoning_btn = gr.Button("Submit")
        reasoning_btn.click(fn=reasoning_text_to_image, inputs=reasoning_txt_input, outputs=[reasoning_img_output, reasoning_text_output])

    with gr.Tab("🖌️ image edit"):
        edit_image_input = gr.Image(label="Input Image")
        edit_prompt = gr.Textbox(label="Prompt")
        edit_image_output = gr.Image(label="Image Generated")
        edit_btn = gr.Button("Submit")
        edit_btn.click(fn=edit_image, inputs=[edit_image_input, edit_prompt], outputs=edit_image_output)

    gr.Markdown("BAGEL-7B-MoT | ByteDance Seed")

demo.launch(server_name="0.0.0.0", server_port=8090)
