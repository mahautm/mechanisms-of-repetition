#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=sf512_OPT
#SBATCH --output=logs/run_OPT_512_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv facebook/opt-1.3b outputs/facebook/opt-1.3b_512_sf --batch-size 8 --use-bnb --max-new-tokens 512 --log-file logs/OPT_512.log
