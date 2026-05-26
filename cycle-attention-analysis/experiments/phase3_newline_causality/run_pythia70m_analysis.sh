#!/bin/bash
#SBATCH --job-name=pythia70m_attention
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pythia70m_attention_%j.out
#SBATCH --error=logs/pythia70m_attention_%j.err

cd /home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality
mkdir -p logs

echo "Starting Pythia-70m attention fallback analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
date

python compare_attention_fallback_alluvial_style.py \
    --model_name EleutherAI/pythia-70m \
    --n_samples 1000 \
    --seed 42

echo "Done!"
date
