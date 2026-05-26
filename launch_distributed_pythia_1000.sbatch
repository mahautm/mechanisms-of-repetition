#!/bin/bash
#SBATCH --job-name=dist_pythia_1000
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --array=0-99%40
#SBATCH --exclude=node044
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dist_pythia_1000_%A_%a.out

set -e

mkdir -p logs

# Total prompts across all ranks
TOTAL_PROMPTS=1000
# Number of array tasks
N_ARRAY=${SLURM_ARRAY_TASK_COUNT:-100}
# Compute per-task prompts (ceil)
PER_TASK=$(( (TOTAL_PROMPTS + N_ARRAY - 1) / N_ARRAY ))

echo "Starting distributed Pythia run: rank ${SLURM_ARRAY_TASK_ID} of ${N_ARRAY}, prompts per task=${PER_TASK}"

source ~/.bashrc
conda activate parr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank ${SLURM_ARRAY_TASK_ID} \
  --n_ranks ${N_ARRAY} \
  --max_prompts ${PER_TASK} \
  --max_new_tokens 512 \
  --batch_size 1 \
  --models "EleutherAI/pythia-1.4b,EleutherAI/pythia-6.9b" \
  --allow_downloads \
  --device cuda
