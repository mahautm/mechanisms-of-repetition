#!/bin/bash
#SBATCH --job-name=simplified_phase1
#SBATCH --output=simplified_phase1_%j.out
#SBATCH --error=simplified_phase1_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4

echo "=== SIMPLIFIED PHASE 1 EXPERIMENTS ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo

# Activate conda environment
source ~/.bashrc
conda activate parr

# Change to project directory
cd /home/mmahaut/projects/parrots/cycle-attention-analysis/src

echo "Python environment:"
which python
python --version
echo

echo "Running simplified Phase 1 experiments..."
python simplified_phase1_experiments.py

echo "Exit code: $?"
echo "End time: $(date)"