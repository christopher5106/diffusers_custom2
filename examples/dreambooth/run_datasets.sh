#!/bin/bash
HOME=/home/ubuntu
mkdir -p MODELS

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


export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"


for DATASET in blonde  bubbleverse 'lineart backdrop' 'newrayman running' sword 'vintage photo' ;
do
  POSTPROMPT="";
  if [ "$DATASET" == "blonde" ];
    then POSTPROMPT=" hollygltly";
  fi;
  if [ "$DATASET" == "newrayman running" ];
    then POSTPROMPT=" rayman";
  fi;

	echo $HOME/$dataset;
  echo $POSTPROMPT;
	python train_dreambooth_lora_sdxl.py \
	--instance_data_dir=$HOME/$dataset \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_prompt="daiton$POSTPROMPT"  \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=0 \
	 --max_train_steps=3000 \
	 --seed=3407 \
	 --lr_scheduler="constant"g \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --mixed_precision="fp16" \
	 --validation_prompt="daiton$POSTPROMPT" \
	 --report_to="wandb"
done;



