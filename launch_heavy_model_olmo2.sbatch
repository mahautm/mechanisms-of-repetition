#!/bin/bash
#SBATCH --job-name=olmo2-multi
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/olmo2_multi_%j.out

set -e

mkdir -p logs

echo "Starting multi-GPU OLMo-2 run"

source ~/.bashrc
conda activate parr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank 0 \
  --n_ranks 1 \
  --max_prompts 1000 \
  --max_new_tokens 1000 \
  --batch_size 4 \
  --models "allenai/OLMo-2-1124-7B" \
  --allow_downloads \
  --device cuda \
  --device_map auto
