#!/bin/bash
#SBATCH --job-name=cycle_table_10rank
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate parr

python compute_cycle_descriptive_stats.py \
  --n_ranks 10 \
  --rank $SLURM_ARRAY_TASK_ID \
  --max_prompts 5000 \
  --prompt_size 512 \
  --max_new_tokens 1000 \
  --batch_size 8 \
  --models EleutherAI/pythia-1.4b \
  --allow_downloads \
  --device cuda \
  --device_map auto
