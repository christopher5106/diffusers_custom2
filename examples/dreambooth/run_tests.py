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

result_dir = Path("results")
result_dir.mkdir(exist_ok=True)

for test in tests:

    dataset = test["dataset_name"]
    print(f"Running tests on dataset {dataset}")

    datasetpath = test["dataset_path"]  # DATASETPATH = " /home/ubuntu/phoenix/data/inputs/tests/train/DATASET/inputs/
    concept_prompt = test["concept_prompt"]
    validation_prompts = test["validation_prompts"]
    to_replace = "sdlfjqlsjlkjmlkjmkljjlmkjlmlkj"
    replacement = ""

    # to_replace = test.get("to_replace", "qsdfkjlqlmdksjflmdqksjflmqjsfmqlksjfm")
    # replacements = test.get("replacements", "")

    for num_special_tokens in [1, 3]:

        if num_special_tokens > 0:
            concept_prompt = ""  # erase concept prompt

        # for replacement in replacements:

        stage1 = {
            "name": "learning special token",
            "params": [
                "--learning_rate", "0",
                "--text_encoder_lr", "0",
                "--text_specialtoken_lr", "1e-4",
                "--max_train_steps", "5",
                "--resume_from_checkpoint", f"MODELS_{rank}/{dataset}/{num_special_tokens}/",
            ]
        }

        stage2 = {
            "name": "full training",
            "params": [
                "--train_text_encoder",
                "--learning_rate", "1e-4",
                "--text_encoder_lr", "1e-6",
                "--text_specialtoken_lr", "1e-6",
                "--resume_from_checkpoint", f"MODELS_{rank}/{dataset}/{num_special_tokens}/",
                "--max_train_steps", "3000",
            ]
        }

        for stage in [stage1, stage2]:
            print(f">TRAINING STAGE: {stage['name']}")
            input_args = [
                "--instance_data_dir", datasetpath,
                "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
                "--output_dir", f"MODELS_{rank}/{dataset}/{num_special_tokens}/",
                "--instance_prompt", concept_prompt,
                "--to_replace", to_replace,
                "--replacement", replacement,
                "--resolution", "1024",
                "--rank", str(rank),
                "--train_batch_size", "1",
                "--gradient_accumulation_steps", "1",
                "--num_special_tokens", str(num_special_tokens),
                "--lr_warmup_steps", "0",
                "--seed", "3407",
                "--lr_scheduler", "constant",
                "--pretrained_vae_model_name_or_path", "madebyollin/sdxl-vae-fp16-fix",
                "--mixed_precision", "fp16",
                "--validation_prompt", concept_prompt,
                "--report_to", "wandb"
            ] + stage["params"]
            args = parse_args(input_args=input_args)

            try:
                with wandb.init(config=vars(args)) as run:
                    train(args)
            except Exception as e:
                print(f"Train error: {e}")
                traceback.print_exc()

        for checkpoint in ["checkpoint-1500", "checkpoint-3000", ""]:  # "checkpoint-500",
            lora_path = f"MODELS_{rank}/{dataset}/{num_special_tokens}/{checkpoint}/pytorch_lora_weights.safetensors"
            _validation_prompts = [
                (concept_prompt + " " + p).lower().replace(to_replace, replacement)
                for p in validation_prompts
            ]
            try:
                generate_lora_sdxl_images(
                    base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
                    lora_path=lora_path,
                    outputs_dir=str(result_dir / dataset / str(num_special_tokens) / checkpoint),
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
