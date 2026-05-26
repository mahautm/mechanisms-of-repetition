#!/bin/bash
#SBATCH --job-name=retry_pythia_100G
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --array=0-99%40
#SBATCH --exclude=node044
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/retry_pythia_100G_%A_%a.out

set -e

mkdir -p logs

RANK=${SLURM_ARRAY_TASK_ID}
N_ARRAY=100
PROMPTS_PER_TASK=10

echo "Retrying rank ${RANK}/${N_ARRAY} with mem=100G"

source ~/.bashrc
conda activate parr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank ${RANK} \
  --n_ranks ${N_ARRAY} \
  --max_prompts ${PROMPTS_PER_TASK} \
  --max_new_tokens 512 \
  --batch_size 2 \
  --models "EleutherAI/pythia-6.9b,allenai/OLMo-2-1124-7B" \
  --allow_downloads \
  --device cuda
