import base64
import requests
from io import BytesIO, StringIO
from typing import Union, Optional, Tuple, List
from PIL import Image, ImageOps
import os
import openai
import json


def get_api_key(file_path):
    # Read the API key from the first line of the file
    with open(file_path, 'r') as file:
        return file.readline().strip()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pick_next_item(current_item, item_list):
    if current_item not in item_list:
        raise ValueError("Current item is not in the list")
    current_index = item_list.index(current_item)
    next_index = (current_index + 1) % len(item_list)

    return item_list[next_index]

# Function to encode a PIL image
def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()
    
    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format='JPEG')
    
    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image


def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """
    Load an image from a given path or URL and convert it to a PIL Image.

    Args:
        image (Union[str, Image.Image]): The image path, URL, or a PIL Image object to be loaded.
        format (str, optional): Desired color format of the resulting image. Defaults to "RGB".
        size (Optional[Tuple], optional): Desired size for resizing the image. Defaults to None.

    Returns:
        Image.Image: A PIL Image in the specified format and size.

    Raises:
        ValueError: If the provided image format is not recognized.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if (size != None):
        image = image.resize(size, Image.LANCZOS)
    return image

class GPT4v():
    def __init__(self, api_key_path='keys/secret.env', are_images_encoded=False, model_name="gpt-4-vision-preview", azure_endpoint=''):
        """OpenAI GPT-4-vision model wrapper
        Args:
            api_key_path (str): Path to the API key file. Defaults to 'keys/secret.env'.
            are_images_encoded (bool): Whether the images are encoded in base64. Defaults to False.
        """
        self.multiple_api_keys = False
        self.current_key_file = None
        self.key_lists = None
        if isinstance(api_key_path, list):
            self.key_lists = api_key_path
            self.current_key_file = api_key_path[0]
            self.api_key = get_api_key(self.current_key_file)
            self.multiple_api_keys = True
        else:
            self.api_key = api_key_path
        
        if not self.api_key:
            print("API key not found.")
            exit(1)

        self.url = azure_endpoint
        self.model_name = model_name
        self.use_encode = are_images_encoded

        self.client = openai.AzureOpenAI(
            azure_endpoint=self.url,
            api_version="2023-07-01-preview",
            api_key=self.api_key
        )

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        prompt_content = []
        text_dict = {
                    "type": "text",
                    "text": text_prompt
                }
        prompt_content.append(text_dict)

        if not isinstance(image_links, list):
            image_links = [image_links]
        for image_link in image_links:
            image = load_image(image_link)
            if self.use_encode == True:
                visual_dict = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(image)}"}
                    }
            else:
                visual_dict = {
                        "type": "image_url",
                        "image_url": {"url": image_link}
                    }
            prompt_content.append(visual_dict)
        return prompt_content

    def get_parsed_output(self, prompt):
        payload = {
            "model": self.model_name,
            "messages": [
            {
                "role": "user",
                "content": prompt
            }
            ],
            "max_tokens": 1400
        }

        response = self.client.chat.completions.create(**payload)
        return self.extract_response(response)
    
    def extract_response(self, response):
        response = json.loads(response.json())

        try:
            out = response['choices'][0]['message']['content']
            return out
        except:
            if response['error']['code'] == 'content_policy_violation':
                print("Code is content_policy_violation")
            elif response['error']['code'] == 'rate_limit_exceeded' or response['error']['code'] == 'insufficient_quota':
                print(f"Code is {response['error']['code']}")
                print(response['error']['message'])
                if self.multiple_api_keys == True:
                    new_key = pick_next_item(self.current_key_file, self.key_lists)
                    self.update_key(new_key)
                    self.current_key_file = new_key #override key
                    print("New key is from the file: ", new_key)
            else:
                print("Code is different")
                print(response)
        return ""

    def update_key(self, key, load_from_file=True):
        if load_from_file:
            self.api_key = get_api_key(key)
        else:
            self.api_key = key

class GPT4o(GPT4v):
    def __init__(self, api_key_path='keys/secret.env', are_images_encoded=False, model_name="gpt-4o-2024-05-13", azure_endpoint=''):
        super().__init__(api_key_path, are_images_encoded, model_name, azure_endpoint=azure_endpoint)

if __name__ == "__main__":
    model = GPT4o('secret_t2.env', model_name="gpt-4.1")
    prompt = model.prepare_prompt(['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print("prompt : \n", prompt)
    res = model.get_parsed_output(prompt)
    print("result : \n", res)