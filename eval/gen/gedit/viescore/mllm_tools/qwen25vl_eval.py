import os
import torch
import time
from PIL import Image
from typing import List
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import requests
from io import BytesIO
import random
import numpy as np
import base64
import magic
import megfile

def process_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def convert_image_to_base64(file_content):
    mime_type = magic.from_buffer(file_content, mime=True)
    base64_encoded_data = base64.b64encode(file_content).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Qwen25VL():
    def __init__(self) -> None:     
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/mnt/jfs-test/pretrained_models/Qwen2.5-VL-72B-Instruct-AWQ", 
            torch_dtype=torch.float16, 
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained("/mnt/jfs-test/pretrained_models/Qwen2.5-VL-72B-Instruct-AWQ")

        print(f"Using {attn_implementation} for attention implementation")

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        image_links_base64 = []

        for img_link in image_links:
            if type(img_link) == str:
                image_links_base64.append(convert_image_to_base64(process_image(megfile.smart_open(img_link, 'rb'))))
            else:
                image_links_base64.append(convert_image_to_base64(process_image(img_link)))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_link} for img_link in image_links_base64
                ] + [{"type": "text", "text": text_prompt}]
            }
        ]
        return messages

    def get_parsed_output(self, messages):
        set_seed(42)
        # Prepare the inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Generate output
        generation_config = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": None,
        }
        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""

if __name__ == "__main__":
    model = Qwen25VL()
    prompt = model.prepare_prompt(
        ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"], 
        'Describe the image in detail.'
    )
    res = model.get_parsed_output(prompt)
    print("result : \n", res)