#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=apertus
#SBATCH --output=logs/apertus_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv swiss-ai/Apertus-8B-2509 outputs/swiss-ai/Apertus-8B-2509_human_lama_parrots_list_v1_sf.csv --batch-size 8 --use-bnb --max-new-tokens 100 --log-file ./logs/apertus_generation.log
