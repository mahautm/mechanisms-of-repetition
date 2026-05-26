#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=sf512_Llama
#SBATCH --output=logs/run_Llama_512_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv meta-llama/Llama-3.2-1B outputs/meta-llama/Llama-3.2-1B_512_sf --batch-size 8 --use-bnb --max-new-tokens 512 --log-file logs/Llama_512.log
