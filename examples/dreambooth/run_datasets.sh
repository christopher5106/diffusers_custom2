#!/bin/bash
HOME=/home/ubuntu
mkdir -p MODELS

python3 train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
  --instance_data_dir="$HOME/phoenix/data/inputs/tests/train/rayman/inputs/" \
  --instance_prompt="daiton" \
  --resolution=1024 \
  --max_train_steps=3000 \
  --seed=3407 \
  --lr_scheduler="constant" \
  --validation_prompt="daiton" \
  --output_dir="MODELS/rayman/" \
  --wandb


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
	--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
	--instance_data_dir="$HOME/datasets/$DATASET/" \
	--output_dir="MODELS/$DATASET/" \
	--instance_prompt="daiton$POSTPROMPT" --resolution=1024 \
	 --max_train_steps=3000 --seed=3407 --lr_scheduler="constant"g \
	 --validation_prompt=daiton --report_to="wandb"
done;



