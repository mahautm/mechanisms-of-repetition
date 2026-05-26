#!/bin/bash
#SBATCH --job-name=multi_pythia_retry
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --array=0-9%5
#SBATCH --exclude=node044
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/multi_pythia_retry_%A_%a.out

set -e

mkdir -p logs

RANK=${SLURM_ARRAY_TASK_ID}
N_ARRAY=10
PROMPTS_PER_TASK=100

echo "Multi-GPU retry rank ${RANK}/${N_ARRAY} (2 GPUs)"

source ~/.bashrc
conda activate parr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank ${RANK} \
  --n_ranks ${N_ARRAY} \
  --max_prompts ${PROMPTS_PER_TASK} \
  --max_new_tokens 512 \
  --batch_size 4 \
  --models "EleutherAI/pythia-6.9b,allenai/OLMo-2-1124-7B" \
  --allow_downloads \
  --device cuda \
  --device_map auto
