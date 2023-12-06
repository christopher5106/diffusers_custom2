#!/bin/bash
HOME=/home/ubuntu
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

mkdir -p TOKEN_SEARCH

DATASET=blonde

for REPLACEMENT in woman '' 'hollygltly' 'hllygltly' 'hllgltl' 'hollygltly woman' 'wmn' 'wmn78610' 'wmnytsqg' 'wmnytsqg woman' 'wmnylsfdqjmkjk' "5876391" "aioaeeua" "holly";
do
  DATASETPATH="$HOME/datasets/$DATASET";
	echo "Dataset: $DATASETPATH";
	echo "replacement: $REPLACEMENT"

  python3 train_dreambooth_lora_sdxl.py \
  --instance_data_dir="$DATASETPATH" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir="TOKEN_SEARCH/$REPLACEMENT/" \
  --instance_prompt="daiton"  \
  --to_replace="woman" \
  --replacement="$REPLACEMENT" \
  --resolution=1024 \
  --rank 4 \
  --train_text_encoder \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --seed=3407 \
  --lr_scheduler="constant" \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --mixed_precision="fp16" \
  --validation_prompt="daiton" \
  --report_to="wandb"

  for CHECKPOINT in "checkpoint-1000" "checkpoint-1500"; # "checkpoint-1500" "checkpoint-5000"
  do
    LORA="TOKEN_SEARCH/$REPLACEMENT/$CHECKPOINT/pytorch_lora_weights.safetensors"
    echo "Loading Loras $LORA"
    python3 predict.py --dataset "$DATASET" \
      --results_dir="results_tokens/$REPLACEMENT/$CHECKPOINT/"  \
      --lora_path="$LORA" \
      --to_replace="woman" \
      --replacement="$REPLACEMENT"

  done

done;



