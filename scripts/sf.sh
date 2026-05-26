#!/bin/bash
source ~/.bashrc
conda activate parr
model=$1
dataset=$2
cd /home/mmahaut/projects/parrots
echo model: $model
echo dataset: $dataset
dataset_name=$(echo $dataset | cut -d'/' -f7 | cut -d'.' -f1)
poetry run python -m parrots.slot_filling \
    $dataset \
    $model \
    "/home/mmahaut/projects/parrots/outputs/${model}_${dataset_name}_sf" \
    --batch-size 64 \
    --max-new-tokens 1000 \
    --log-file /home/mmahaut/projects/parrots/logs/${model}_${dataset_name}_1000sf.log \
    --use-bnb \
    --no-use-accelerator \


