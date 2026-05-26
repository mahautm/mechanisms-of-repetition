#!/bin/bash
#SBATCH --job-name=compute_pythia_targets
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/compute_pythia_targets_%j.out

set -e

mkdir -p logs

echo "Starting targeted Pythia run..."

source ~/.bashrc
conda activate parr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank 0 \
  --n_ranks 1 \
  --max_prompts 1000 \
  --max_new_tokens 512 \
  --batch_size 1 \
  --models "EleutherAI/pythia-1.4b,EleutherAI/pythia-6.9b" \
  --allow_downloads \
  --device cuda
