#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=m_OLMo_icl
#SBATCH --output=logs/run_OLMo_icl_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv allenai/OLMo-1B-hf outputs/mitigations/OLMo_icl --batch-size 8 --use-bnb --max-new-tokens 512 --use-icl --log-file logs/OLMo_icl.log
