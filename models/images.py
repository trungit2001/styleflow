import random

from PIL import Image
from models.utils import read_text


def image_loader(image_path) -> Image:
    return Image.open(image_path).convert("RGB")


def get_data_shuffle(content_file, style_file) -> dict: 
    content_images = read_text(content_file)
    style_images = read_text(style_file)

    tmp_data = list(zip(content_images, style_images))
    random.shuffle(tmp_data)
    content_images_shuffled, style_images_shuffled = zip(*tmp_data)

    return {
        "content_images_shuffled": content_images_shuffled,
        "style_image_shuffled": style_images_shuffled
    }
