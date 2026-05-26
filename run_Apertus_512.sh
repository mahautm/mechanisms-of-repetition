#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=sf512_Apertus
#SBATCH --output=logs/run_Apertus_512_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv swiss-ai/Apertus-8B-2509 outputs/swiss-ai/Apertus-8B-2509_512_sf --batch-size 8 --use-bnb --max-new-tokens 512 --log-file logs/Apertus_512.log
