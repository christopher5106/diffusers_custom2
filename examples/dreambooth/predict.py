"""Script used to make inferences with a LORA-trained model using SDXL (without refiner)"""
import os
import random
import argparse

from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import json
from PIL import Image
import math

from diffusers import StableDiffusionXLPipeline
from loguru import logger

# fix seed
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)  # noqa: NPY002
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def generate_lora_sdxl_images(
    *,
    base_model_path: str,
    lora_path: str,
    outputs_dir: str,
    prompts,
    num_images: int,
    num_inference_steps: int,
    prompts_2: str = None,
) -> None:
    """Generate images from a text prompt using LORA weights and sdxl.

    Args:
        base_model_path (str): the path or repo id to the base model
        lora_path (str): the path to the lora weights
        outputs_dir (str): the path to the directory where the generated images will be saved
        prompts (str | Iterable[str]): the text prompt to generate images from
        num_images (int): the number of images to generate
        num_inference_steps (int): the number of inference steps to use
        prompts_2 (str | Iterable[str] | None): the 2nd text prompt to generate images from

    For info on the 2nd prompt, see https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#use-a-different-prompt-for-each-text-encoder
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        Path.mkdir(outputs_dir, parents=True)

    logger.info(f"Loading model from {base_model_path}")
    model = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
    )
    logger.info(f"Loading LORA weights from {lora_path}")
    model.load_lora_weights(
        lora_path,
    )  # beware, vscode points to LoraLoaderMixin instead of StableDiffusionXLLoraLoaderMixin
    # or model.unet.load_attn_procs(lora_path)
    logger.info("Moving model to GPU")
    model.to("cuda")  # move pipe to GPU

    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(prompts_2, str) or prompts_2 is None:
        prompts_2 = [prompts_2]
    if len(prompts) != len(prompts_2):
        logger.warning(
            f"Number of prompts ({len(prompts)}) and number of prompts_2 ({len(prompts_2)}) are different; proceeding padding with None",
        )
    for prompt, prompt2 in zip_longest(prompts, prompts_2):
        generated_images = []

        for ind in range(num_images):
            logger.info(
                f"Generating image {ind+1} from prompt: {prompt} (prompt 1){(' and ' + prompt2+' (prompt 2)') if prompt2 else ''}",
            )
            image = model(  # check class doc for all params
                prompt=prompt,
                num_inference_steps=num_inference_steps,
            ).images[0]
            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            # filename = outputs_dir / (prompt[:100]) / f"lora_sdxl_{ind}.png"
        # (outputs_dir / (prompt[:100])).mkdir(exist_ok=True, parents=True)
            # logger.info(f"Saving image to {filename}")
            # image.save(filename)
            generated_images.append(image)

        image = image_grid(generated_images, 2, math.ceil(len(generated_images) / 2))
        filename = outputs_dir / (str(prompt[:100]) + ".png")
        image.save(filename)
        logger.info(f"Saving image to {filename}")

    del model
    torch.cuda.empty_cache()


with open("prompts.json", "r") as f:
    prompts = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple prediction script.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )
    parser.add_argument(
        "--to_replace",
        type=str,
        default="qsdfkjlqlmdksjflmdqksjflmqjsfmqlksjfm",
    )
    parser.add_argument(
        "--replacement",
        type=str,
        default="",
    )
    args = parser.parse_args()

    _prompts = prompts[args.dataset]["prompts"]
    concept_prompt = prompts[args.dataset]["concept_prompt"]

    generate_lora_sdxl_images(
        base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
        lora_path=args.lora_path,
        outputs_dir=args.results_dir,
        prompts=[(concept_prompt + " " + p).replace(args.to_replace, args.replacement) for p in _prompts],
        num_images=10,
        num_inference_steps=30,
    )
