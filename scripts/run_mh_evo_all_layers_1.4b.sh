#!/bin/bash
#SBATCH --job-name=mh_evo_1.4b
#SBATCH --output=outputs_multihead_full/logs/mh_evo_1.4b_L%a.out
#SBATCH --error=outputs_multihead_full/logs/mh_evo_1.4b_L%a.err
#SBATCH --mem=64G
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-23

# Pythia-1.4b has 24 layers (0-23)
LAYER=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0
cd ~/projects/parrots
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running layer $LAYER"
poetry run python multihead_original_vs_acquired_evolution.py \
    --model-name=EleutherAI/pythia-1.4b \
    --layer=$LAYER \
    --n-samples=300 \
    --batch-size=1

echo "Done with layer $LAYER"
