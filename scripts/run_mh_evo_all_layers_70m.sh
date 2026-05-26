#!/bin/bash
#SBATCH --job-name=mh_evo_70m
#SBATCH --output=outputs_multihead_full/logs/mh_evo_70m_L%a.out
#SBATCH --error=outputs_multihead_full/logs/mh_evo_70m_L%a.err
#SBATCH --mem=64G
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-5

# Pythia-70m has 6 layers (0-5)
LAYER=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0
cd ~/projects/parrots
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running layer $LAYER"
poetry run python multihead_original_vs_acquired_evolution.py \
    --model-name=EleutherAI/pythia-70m \
    --layer=$LAYER \
    --n-samples=300 \
    --batch-size=8

echo "Done with layer $LAYER"
