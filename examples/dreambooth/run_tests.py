import argparse
from train_dreambooth_lora_sdxl import main as train
from train_dreambooth_lora_sdxl import parse_args as train_parse_args
from predict import parse_args as predict_parse_args

with open("tests.json", "r") as f:
    tests = json.load(f)

rank = 4

for test in tests:

    dataset = test["dataset_name"]
    datasetpath = test["dataset_path"]  # DATASETPATH = " /home/ubuntu/phoenix/data/inputs/tests/train/DATASET/inputs/
    concept_prompt = test["concept_prompt"]
    validation_prompts = test["validation_prompts"]
    to_replace = test.get("to_replace", "qsdfkjlqlmdksjflmdqksjflmqjsfmqlksjfm")
    replacements = test.get("replacement", "")

    for replacement in replacements:

        args = train_parse_args([
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
            "--lr_warmup_steps", "0",
            "--max_train_steps", "1500",
            "--seed", "3407",
            "--lr_scheduler", "constant",
            "--pretrained_vae_model_name_or_path", "madebyollin/sdxl-vae-fp16-fix",
            "--mixed_precision", "fp16",
            "--validation_prompt", "daiton",
            "--report_to", "wandb"
        ])

        try:
            train(args)
        except Exception as e:
            print(f"Train error: {e}")

        for checkpoint in ["checkpoint-1500"]:
            lora_path = f"MODELS_{rank}/{dataset}/{replacement}/{checkpoint}/pytorch_lora_weights.safetensors"
            try:
                generate_lora_sdxl_images(
                    base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
                    lora_path=lora_path,
                    outputs_dir="results",
                    prompts=[(concept_prompt + " " + p).replace(to_replace, replacement) for p in validation_prompts],
                    num_images=10,
                    num_inference_steps=30,
                )
            except Exception as e:
                print(f"Inference error {e}")
