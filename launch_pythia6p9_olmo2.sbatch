#!/bin/bash
#SBATCH --job-name=pythia6p9_olmo2
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/pythia6p9_olmo2_%j.out

set -e

mkdir -p logs

echo "Starting targeted run: pythia-6.9b and OLMo-2..."

source ~/.bashrc
conda activate parr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank 0 \
  --n_ranks 1 \
  --max_prompts 1000 \
  --max_new_tokens 512 \
  --batch_size 1 \
  --models "EleutherAI/pythia-6.9b,allenai/OLMo-2-1124-7B" \
  --allow_downloads \
  --device cuda
