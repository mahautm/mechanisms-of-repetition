#!/bin/bash
#SBATCH --job-name=apertus_sf
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=apertus_sf_%j.out
#SBATCH --error=apertus_sf_%j.err

source ~/.bashrc
conda activate parr

export HF_HOME=/gpfs/scratch/$USER/huggingface_cache

python parrots/slot_filling.py \
    data/human_lama_parrots_list_v1.csv \
    swiss-ai/Apertus-8B-2509 \
    outputs/ \
    --batch-size 8 \
    --max-new-tokens 50 \
    --log-file logs/Apertus-8B-2509_sf.log

