import argparse
from train_dreambooth_lora_sdxl import main as train
from train_dreambooth_lora_sdxl import parse_args

DATASET="blonde"
RANK=4
DATASETPATH = f"/home/ubuntu/phoenix/data/inputs/tests/train/{DATASET}/inputs/"
# DATASETPATH = "/home/ubuntu/datasets/$DATASET";

args = parse_args()
args.instance_data_dir=DATASETPATH
args.pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
args.output_dir=f"MODELS_{RANK}/{DATASET}"
args.instance_prompt="daiton"
args.resolution=1024
args.rank=RANK
args.train_text_encoder=True
args.train_batch_size=1
args.gradient_accumulation_steps=1
args.learning_rate=1e-4
args.lr_warmup_steps=0
args.max_train_steps=1500
args.seed=3407
args.lr_scheduler="constant"
args.pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"
args.mixed_precision="fp16"
args.validation_prompt="daiton"
args.report_to="wandb"


print(args.train_text_encoder, args.resolution, args.rank)
main(args)
