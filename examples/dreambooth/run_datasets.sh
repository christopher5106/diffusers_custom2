#!/bin/bash
HOME=/home/ubuntu
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

#python3 train_dreambooth_lora_sdxl.py \
#  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
#  --instance_data_dir="$HOME/phoenix/data/inputs/tests/train/rayman/inputs/" \
#  --instance_prompt="daiton" \
#  --resolution=1024 \
#  --max_train_steps=3000 \
#  --seed=3407 \
#  --lr_scheduler="constant" \
#  --validation_prompt="daiton" \
#  --output_dir="MODELS/rayman/" \
#  --wandb

RANK=4
mkdir -p MODELS_$RANK
#for DATASET in blonde  bubbleverse 'lineart backdrop' 'newrayman running' sword 'vintage photo' rayman3;
for DATASET in rayman3;
do
  POSTPROMPT="";
  if [ "$DATASET" = "blonde" ];
    then POSTPROMPT=" hollygltly";
  fi;
  if [ "$DATASET" = "newrayman running" ];
    then POSTPROMPT=" rayman";
  fi;

	echo "Dataset: $HOME/datasets/$DATASET";
  echo $POSTPROMPT;

#  python3 train_dreambooth_lora_sdxl.py \
#  --instance_data_dir="$HOME/datasets/$DATASET" \
#  --pretrained_model_name_or_path=$MODEL_NAME  \
#  --output_dir="MODELS_$RANK/$DATASET/" \
#  --instance_prompt="daiton$POSTPROMPT"  \
#  --resolution=1024 \
#  --rank $RANK \
#  --train_text_encoder \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=1 \
#  --learning_rate=1e-4 \
#  --lr_warmup_steps=0 \
#  --max_train_steps=3000 \
#  --seed=3407 \
#  --lr_scheduler="constant" \
#  --pretrained_vae_model_name_or_path=$VAE_PATH \
#  --mixed_precision="fp16" \
#  --validation_prompt="daiton$POSTPROMPT" \
#  --report_to="wandb"


  for CHECKPOINT in "checkpoint-1500", "checkpoint-3000"; # checkpoint-5000
  do
    python3 predict.py --dataset "$DATASET" \
      --results_dir="results_$RANK/$DATASET/$CHECKPOINT"  \
      --lora_path="MODELS_$RANK/$DATASET/$CHECKPOINT/pytorch_lora_weights.safetensors"
  done

done;



