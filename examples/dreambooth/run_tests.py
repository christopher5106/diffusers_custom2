import argparse
import traceback
import json
from pathlib import Path
import wandb

from train_dreambooth_lora_sdxl import main as train
from train_dreambooth_lora_sdxl import parse_args
from predict import generate_lora_sdxl_images

with open("tests.json", "r") as f:
    tests = json.load(f)

rank = 64
num_steps = 1500
num_special_tokens = 1

result_dir = Path("results")
result_dir.mkdir(exist_ok=True)

for test in tests:

    dataset = test["dataset_name"]
    print(f"Running tests on dataset {dataset}")

    datasetpath = test["dataset_path"]  # DATASETPATH = " /home/ubuntu/phoenix/data/inputs/tests/train/DATASET/inputs/
    concept_prompt = test["concept_prompt"]
    validation_prompts = test["validation_prompts"]
    to_replace = "sdlfjqlsjl kjmlk jm kljj lmkjl mlkj "
    replacement = ""

    # to_replace = test.get("to_replace", "qsdfkjlqlmdksjflmdqksjflmqjsfmqlksjfm")
    # replacements = test.get("replacements", "")

    if num_special_tokens > 0:
        concept_prompt = " "  # erase concept prompt

    # for replacement in replacements:

    print(f"  replacement of {to_replace}: {replacement}")
    input_args = [
        "--instance_data_dir", datasetpath,
        "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
        "--output_dir", f"MODELS_{rank}/{dataset}/{replacement}",
        "--instance_prompt", concept_prompt,
        "--to_replace", to_replace,
        "--replacement", replacement,
        "--resolution", "1024",
        "--rank", str(rank),
        "--train_text_encoder",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "1e-4",
        "--num_special_tokens", str(num_special_tokens),
        "--text_specialtoken_lr", "1e-3",
        "--text_encoder_lr", "1e-6",
        "--lr_warmup_steps", "0",
        "--max_train_steps", f"{num_steps}",
        "--seed", "3407",
        "--lr_scheduler", "constant",
        "--pretrained_vae_model_name_or_path", "madebyollin/sdxl-vae-fp16-fix",
        "--mixed_precision", "fp16",
        "--validation_prompt", concept_prompt,
        "--report_to", "wandb"
    ]
    args = parse_args(input_args=input_args)

    try:
        with wandb.init(config=vars(args)) as run:
            train(args)
    except Exception as e:
        print(f"Train error: {e}")
        traceback.print_exc()

    for checkpoint in ["checkpoint-500", "checkpoint-1000", ""]:
        lora_path = f"MODELS_{rank}/{dataset}/{replacement}/{checkpoint}/pytorch_lora_weights.safetensors"
        _validation_prompts = [
            (concept_prompt + " " + p).lower().replace(to_replace, replacement)
            for p in validation_prompts
        ]
        try:
            generate_lora_sdxl_images(
                base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
                lora_path=lora_path,
                outputs_dir=str(result_dir / dataset / checkpoint),
                prompts=_validation_prompts,
                num_images=10,
                num_inference_steps=30,
                num_special_tokens=num_special_tokens
            )
        except Exception as e:
            print(f"Inference error {e}")
            traceback.print_exc()


#     html = f"<h1>{dataset}</h1>"
#
#     for prompt in validation_prompts:
#
#         html += f"<h4>{prompt}</h4>"
#
#         for replacement in replacements:
#             p = (concept_prompt + " " + prompt).lower().replace(to_replace, replacement)
#             html += "<p>" + p + "</p>"
#             gridimage_path = dataset + "/" + (str(p[:100]).replace(".", "") + ".png")
#             html += f"<img src='{gridimage_path}' width='100%'/><br/>"
#
#     with open(result_dir / f"{dataset}.html", "w") as f:
#         f.write(html)
#
#
# html = f"<h1>Train test results</h1><ul>"
# for test in tests:
#     dataset = test["dataset_name"]
#     html += f"<li><a href='{dataset}.html'>{dataset}</a></li>"
# html += "</ul>"
# with open(result_dir / f"index.html", "w") as f:
#     f.write(html)

#  aws s3 cp --recursive results s3://dev-ml-phoenix-test-reports-bucket83908e77-1reu5qi5chdwa/training/2023-12-08-tokens-variation/results
