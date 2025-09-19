import os
import math
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def dynamic_preprocess(image: Image.Image,
                       image_size: int = 448,
                       max_num: int = 12,
                       use_thumbnail: bool = False):

    w, h = image.size
    processed_images = []


    if min(w, h) <= image_size:
        processed_images.append(image.resize((image_size, image_size), resample=Image.BICUBIC))

    else:

        n = max(1, int(max_num))  # 防御

        x_edges = [round(i * w / n) for i in range(n + 1)]
        y_edges = [round(j * h / n) for j in range(n + 1)]


        for gy in range(n):
            for gx in range(n):
                left, top = x_edges[gx], y_edges[gy]
                right, bottom = x_edges[gx + 1], y_edges[gy + 1]
                tile = image.crop((left, top, right, bottom))
                tile = tile.resize((image_size, image_size), resample=Image.BICUBIC)
                processed_images.append(tile)


    if use_thumbnail and len(processed_images) > 1:
        thumb = image.resize((image_size, image_size), resample=Image.BICUBIC)
        processed_images.append(thumb)

    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12):

    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)

    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values, dim=0)  # [num_views, 3, H, W]
    return pixel_values


def generate_captions(input_folder: str,
                      output_folder: str,
                      model,
                      tokenizer,
                      max_num: int = 12):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir in os.listdir(input_folder):
        sub_input_path = os.path.join(input_folder, subdir)
        sub_output_path = os.path.join(output_folder, subdir)

        if os.path.isdir(sub_input_path):
            os.makedirs(sub_output_path, exist_ok=True)

            for image_name in os.listdir(sub_input_path):
                image_path = os.path.join(sub_input_path, image_name)
                name, ext = os.path.splitext(image_name)


                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue

                output_path = os.path.join(sub_output_path, name + '.txt')

                if os.path.exists(output_path):
                    continue

                try:
                    pixel_values = load_image(image_path, input_size=448, max_num=max_num).to(torch.bfloat16).cuda()
                    question = (
                        "<image>\n"
                        "This is a hematoxylin and eosin stained histopathological image of gastric cancer tissue. Please provide a description of the overall tissue."
                    )
                    generation_config = dict(max_new_tokens=1024, do_sample=True)


                    with torch.inference_mode():
                        response = model.chat(tokenizer, pixel_values, question, generation_config)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(response)

                    print(f"Generated caption for {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")




path = r"\InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

input_folder = r"\image"
output_folder = r"\txt"

generate_captions(input_folder, output_folder, model, tokenizer, max_num=12)
