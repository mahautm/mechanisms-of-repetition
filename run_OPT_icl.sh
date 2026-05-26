#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=m_OPT_icl
#SBATCH --output=logs/run_OPT_icl_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv facebook/opt-1.3b outputs/mitigations/OPT_icl --batch-size 8 --use-bnb --max-new-tokens 512 --use-icl --log-file logs/OPT_icl.log
