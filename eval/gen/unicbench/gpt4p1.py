# -*- coding: utf-8 -*-

import argparse
import requests
import time
import json
import hmac
import hashlib
import base64
from typing import Union, Optional, Tuple, List
from io import BytesIO

def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()
    
    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format='JPEG')
    
    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image

class GPT4o:
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-2025-04-14"):
        """OpenAI GPT-4-vision model wrapper
        Args:
            api_key (str): API key for authentication.
            are_images_encoded (bool): Whether the images are encoded in base64. Defaults to False.
        """
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"
        self.model_name = model_name

    def forward(self, final_prompt, max_retries=10):
        for retry in range(max_retries):
            response = self.get_parsed_output(final_prompt)
            if response is not None:
                return response
            else:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    print(f"Failed to get response (attempt {retry + 1}/{max_retries})")
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

    def prepare_prompt(self, image_list: List = [], text_prompt: str = ""):
        if image_list is None:
            print("img_base64_list is None")
            return None
        else:
            if not isinstance(image_list, list):
                image_list = [image_list]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                    ] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(img_base64)}"}} for img_base64 in image_list
                    ]
                }
            ]
            return messages
        
    def get_parsed_output(self, prompt):
        payload = {
            "model": self.model_name,
            "temperature": 0.1,
            "messages": prompt,
            "max_tokens": 2048,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            r = requests.post(self.url, json=payload, headers=headers)
            r_content = r.content
            res_json = r.json()
        except Exception as e:
            print(f"Error Occur, content: {r_content}, exception: {str(e)}")
            res_json = None

        msg = res_json.get('msg', None)
        if msg is None or msg != 'success':
            print(f"Error Occur, msg: {msg}")
            return None
        return res_json['response']