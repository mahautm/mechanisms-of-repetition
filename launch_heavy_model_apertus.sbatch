#!/bin/bash
#SBATCH --job-name=apertus-8b-multi
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/apertus8b_multi_%j.out

set -e

mkdir -p logs

echo "Starting multi-GPU Apertus-8B run"

source ~/.bashrc
conda activate parr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank 0 \
  --n_ranks 1 \
  --max_prompts 1000 \
  --max_new_tokens 1000 \
  --batch_size 4 \
  --models "swiss-ai/Apertus-8B-2509" \
  --allow_downloads \
  --device cuda \
  --device_map auto
