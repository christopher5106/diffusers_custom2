import argparse
from train_dreambooth_lora_sdxl import main as train
from train_dreambooth_lora_sdxl import parse_args

DATASET="blonde"
RANK=4
DATASETPATH = f"/home/ubuntu/phoenix/data/inputs/tests/train/{DATASET}/inputs/"
# DATASETPATH = "/home/ubuntu/datasets/$DATASET";

args = parse_args([
    "--instance_data_dir", DATASETPATH,
    "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
    "--output_dir", f"MODELS_{RANK}/{DATASET}",
    "--instance_prompt", "daiton",
    "--resolution", "1024",
    "--rank", str(RANK),
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

print(args.train_text_encoder, args.resolution, args.rank)
main(args)
