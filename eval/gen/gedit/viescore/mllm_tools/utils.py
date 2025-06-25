from typing import List
import base64
from io import BytesIO
from PIL import Image
import requests

def pil_image_to_base64(pil_image, format="PNG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)  # Save image to the buffer in the specified format
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Encode the buffer's content to base64
    return img_str

def load_image(image_file):
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        import os
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def merge_images(image_links: List = []):
        """Merge multiple images into one image

        Args:
            image_links (List, optional): List of image links. Defaults to [].

        Returns:
            [type]: [description]
        """
        if len(image_links) == 0:
            return None
        images = load_images(image_links)
        if len(images) == 1:
            return images[0]
        widths, heights = zip(*(i.size for i in images))
        average_height = sum(heights) // len(heights)
        for i, im in enumerate(images):
            # scale in proportion
            images[i] = im.resize((int(im.size[0] * average_height / im.size[1]), average_height))
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new("RGB", (total_width + 10 * (len(images) - 1), max_height))
        x_offset = 0
        for i, im in enumerate(images):
            if i > 0:
                # past a column of 1 pixel starting from x_offset width being black, 8 pixels being white, and 1 pixel being black
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
                new_im.paste(Image.new("RGB", (8, max_height), (255, 255, 255)), (x_offset, 0))
                x_offset += 8
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im