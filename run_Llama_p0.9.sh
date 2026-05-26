#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=m_Llama_p0.9
#SBATCH --output=logs/run_Llama_p0.9_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv meta-llama/Llama-3.2-1B outputs/mitigations/Llama_p0.9 --batch-size 8 --use-bnb --max-new-tokens 512 --top-p 0.9 --log-file logs/Llama_p0.9.log
