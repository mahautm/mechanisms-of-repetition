#!/bin/bash
#SBATCH --job-name=dist_small_models
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --array=0-99%40
#SBATCH --exclude=node044
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dist_small_models_%A_%a.out

set -e

mkdir -p logs

MODELS=("EleutherAI/pythia-70m" "EleutherAI/pythia-1.4b" "meta-llama/Llama-3.2-1B" "allenai/OLMo-1B-hf")
N_MODELS=${#MODELS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$(( TASK_ID % N_MODELS ))
MODEL=${MODELS[$MODEL_IDX]}

TOTAL_PROMPTS=1000
N_ARRAY=${SLURM_ARRAY_TASK_COUNT:-100}
PER_TASK=$(( (TOTAL_PROMPTS + N_ARRAY - 1) / N_ARRAY ))

echo "Rank ${TASK_ID}/${N_ARRAY} -> Model ${MODEL}, prompts=${PER_TASK}"

source ~/.bashrc
conda activate parr

python /home/mmahaut/projects/parrots/compute_cycle_descriptive_stats.py \
  --rank ${TASK_ID} \
  --n_ranks ${N_ARRAY} \
  --max_prompts ${PER_TASK} \
  --max_new_tokens 1000 \
  --batch_size 2 \
  --models "${MODEL}" \
  --allow_downloads \
  --device cuda
