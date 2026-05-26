#!/bin/bash
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --job-name=cycle_grid
#SBATCH --output=logs/cycle_grid_chunks_%A_%a.out
#SBATCH --error=logs/cycle_grid_chunks_%A_%a.err

source ~/.bashrc
conda activate parr

# 6 checkpoints x 3 chunks = 18 tasks
CHECKPOINTS=("step1" "step1000" "step5000" "step10000" "step100000" "steplatest")
CHUNKS=3
CHUNK_SIZE=100
TOTAL_SAMPLES=300

CHECKPOINT_INDEX=$((SLURM_ARRAY_TASK_ID / CHUNKS))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % CHUNKS))
CHECKPOINT=${CHECKPOINTS[$CHECKPOINT_INDEX]}

if [ -z "$CHECKPOINT" ]; then
  echo "Invalid checkpoint index: $CHECKPOINT_INDEX"
  exit 1
fi

echo "Running checkpoint: $CHECKPOINT | chunk $CHUNK_INDEX/$CHUNKS"

time python multihead_original_vs_acquired_evolution.py \
  --model-name=EleutherAI/pythia-70m \
  --all-layers \
  --checkpoint=$CHECKPOINT \
  --n-cycle-iterations=6 \
  --n-samples=$CHUNK_SIZE \
  --total-samples=$TOTAL_SAMPLES \
  --chunk-index=$CHUNK_INDEX \
  --batch-size=8 \
  --max-new-tokens=1000

echo "Done with $CHECKPOINT chunk $CHUNK_INDEX"
