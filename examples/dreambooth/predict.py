"""Script used to make inferences with a LORA-trained model using SDXL (without refiner)"""
import os
import random

from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import json

from diffusers import StableDiffusionXLPipeline
from loguru import logger

# fix seed
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)  # noqa: NPY002
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
TOKEN = "daiton"  # noqa: S105

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
    if not Path(outputs_dir).exists():
        Path.mkdir(Path(outputs_dir), parents=True)

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
        os.makedirs(Path(outputs_dir) / (prompt[:100]), exist_ok=True)
        for ind in range(num_images):
            logger.info(
                f"Generating image {ind+1} from prompt: {prompt} (prompt 1){(' and ' + prompt2+' (prompt 2)') if prompt2 else ''}",
            )
            image = model(  # check class doc for all params
                prompt=prompt,
                num_inference_steps=num_inference_steps,
            ).images[0]
            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            filename = Path(outputs_dir) / (prompt[:100]) / f"lora_sdxl_{ind}.png"
            logger.info(f"Saving image to {filename}")
            image.save(filename)

    del model
    torch.cuda.empty_cache()


with open("prompts.json", "r") as f:
    prompts = json.load(f)

if __name__ == "__main__":

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for dataset in ['blonde', 'bubbleverse', 'lineart backdrop', 'newrayman running',
                    'sword', 'vintage photo']:
        postprompt = ""
        if dataset == "newrayman running":
            postprompt = " rayman"

        _prompts = prompts[dataset]
        for checkpoint in ["checkpoint-1500", "checkpoint-3000"]:
            generate_lora_sdxl_images(
                base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
                lora_path=f"MODELS/{dataset}/{checkpoint}/pytorch_lora_weights.safetensors",
                outputs_dir=f"{results_dir}/{dataset}/{checkpoint}",
                prompts=["daiton" + postprompt + " " + p for p in _prompts],
                num_images=10,
                num_inference_steps=30,
            )
