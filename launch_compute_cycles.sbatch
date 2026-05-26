#!/bin/bash
#SBATCH --job-name=compute_cycles
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --array=0-9%10
#SBATCH --exclude=node044
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/compute_cycles_%A_%a.out

set -e

N_ARRAY=10
mkdir -p logs

echo "Starting rank ${SLURM_ARRAY_TASK_ID} of ${N_ARRAY}..."

source ~/.bashrc
conda activate parr

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
    --rank ${SLURM_ARRAY_TASK_ID} \
    --n_ranks ${N_ARRAY} \
    --max_prompts 1000 \
    --max_new_tokens 512 \
    --batch_size 1 \
    --allow_downloads \
    --device cuda
