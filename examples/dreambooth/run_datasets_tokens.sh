#!/bin/bash
HOME=/home/ubuntu
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

mkdir -p TOKEN_SEARCH

DATASET=blonde

#coffee_cup
#cup => coffee cup
#cup => cffcupp
#
#        "A coffee cup steaming on a kitchen counter.",
#        "Coffee cup left on an office desk.",
#        "Empty coffee cup sitting in a sink.",
#        "Watercolor painting of a coffee cup",
#        "Coffee cup spilled on a table.",
#        "Coffee cup on a bedside table.",
#        "Antique coffee cup in a display case.",
#        "A charcoal sketch of a coffee cup"

# "astarion"
#man => ""
#Astrn
#Astrn man
#
#    • Man standing in a grand hall.
#    • Man overlooking a vast estate from a balcony.
#    • Man seated at a luxurious feast.
#    • Man with a thoughtful expression holding an ancient tome.
#    • Man in a duel wielding a rapier.
#    • Man on horseback riding through the countryside.
#    • Man negotiating at a political gathering.
#    • Man studying a map sprawled across a table.
#    • Man receiving guests in a lavish drawing room.
#    • Oil painted portrait of a man


# confidential_mushroom_lady
# mshrmcrtr
#mshrmcrtr woman

#"woman lurking in a forest.",
#                                "woman sitting on a log.",
#                                "woman under a full moon.",
#                                "woman with glowing spots.",
#                                "woman in a fairy ring.",
#                                "woman hiding behind ancient ruins.",
#                                "woman with a mischievous grin.",
#                                "woman standing in a mystical swamp.",
#                                "woman in a wizard's garden.",
#                                "woman being sketched by a naturalist."

#illustrated_woman
#mmkrlssn
#mmkrlssn woman
#                "mmkrlssn woman reading a book in a park.",
#                "mmkrlssn woman gazing out of a window on a rainy day.",
#                "mmkrlssn woman walking through a bustling city street.",
#                "mmrklssn woman with a mysterious smile holding a flower.",
#                "mmkrlssn woman dancing under the moonlight.",
#                "mmkrlssn woman painting on a large canvas.",
#                "mmkrlssn woman sipping coffee in a café.",
#                "mmkrlssn woman running through a field of wildflowers.",
#                "mmkrlssn woman in a flowing dress at a masquerade ball.",
#                "mmkrlssn woman sitting beside a tranquil lake."


pink_diamond_sword
pnkdmnd
pnkdmnd sword
#        "Sword gleaming in the morning sun.",
#        "Sword stuck in a stone.",
#        "Sword laying on a pile of gold coins.",
#        "Sword with intricate engravings.",
#        "Sword cast aside on a battlefield.",
#        "Sword with a jeweled hilt.",
#        "Sword displayed on a wall.",
#        "Rusty sword in an old chest.",
#        "Sword wrapped in a royal banner.",
#        "Sword being forged by a blacksmith."

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



