#!/bin/bash
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --job-name=cycle_grid
#SBATCH --output=logs/cycle_grid_array_%A_%a.out
#SBATCH --error=logs/cycle_grid_array_%A_%a.err

source ~/.bashrc
conda activate parr

# Map array index to checkpoint
CHECKPOINTS=("step1" "step1000" "step5000" "step10000" "step100000" "steplatest")
CHECKPOINT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

echo "Running checkpoint: $CHECKPOINT (array task $SLURM_ARRAY_TASK_ID)"

python multihead_original_vs_acquired_evolution.py \
    --model-name=EleutherAI/pythia-70m \
    --all-layers \
    --checkpoint=$CHECKPOINT \
    --n-cycle-iterations=6 \
    --n-samples=300 \
    --batch-size=8 \
    --max-new-tokens=1000

echo "Done with $CHECKPOINT"
