#!/bin/bash
#SBATCH --job-name=phase1_aggressive_experiments
#SBATCH --output=phase1_aggressive_experiments_%j.out
#SBATCH --error=phase1_aggressive_experiments_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

echo "=== PHASE 1: AGGRESSIVE REPETITION INDUCTION EXPERIMENTS ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo

# Source bashrc and activate conda environment
source ~/.bashrc
conda activate parr

# Change to project directory
cd /home/mmahaut/projects/parrots/cycle-attention-analysis/src

echo "Python environment:"
which python
python --version
echo

# Create output directory for all Phase 1 results
mkdir -p /home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/phase1_results
echo "Created Phase 1 results directory"

echo "=== EXPERIMENT 1: GRADIENT-BASED REPETITION OPTIMIZATION ==="
echo "Start time: $(date)"
echo "Testing gradient ascent to maximize cycle detection scores..."
python gradient_based_repetition_experiment.py
gradient_exit_code=$?
echo "Gradient-based experiment completed with exit code: $gradient_exit_code"
echo "End time: $(date)"
echo

echo "=== EXPERIMENT 2: DIRECT EMBEDDING MANIPULATION ==="
echo "Start time: $(date)"
echo "Testing direct token embedding manipulation..."
python embedding_manipulation_experiment.py
embedding_exit_code=$?
echo "Embedding manipulation experiment completed with exit code: $embedding_exit_code"
echo "End time: $(date)"
echo

echo "=== EXPERIMENT 3: RESIDUAL STREAM INTERRUPTION ==="
echo "Start time: $(date)"
echo "Testing residual stream interruption hooks..."
python residual_interruption_experiment.py
residual_exit_code=$?
echo "Residual interruption experiment completed with exit code: $residual_exit_code"
echo "End time: $(date)"
echo

echo "=== PHASE 1 SUMMARY ==="
echo "Gradient-based exit code: $gradient_exit_code"
echo "Embedding manipulation exit code: $embedding_exit_code"
echo "Residual interruption exit code: $residual_exit_code"

if [ $gradient_exit_code -eq 0 ] && [ $embedding_exit_code -eq 0 ] && [ $residual_exit_code -eq 0 ]; then
    echo "✅ ALL PHASE 1 EXPERIMENTS COMPLETED SUCCESSFULLY!"
else
    echo "❌ Some experiments failed. Check individual logs."
fi

echo
echo "Total job end time: $(date)"
echo "Results saved in individual experiment directories:"
echo "- Gradient-based: plots/gradient_based_experiment/"
echo "- Embedding manipulation: plots/embedding_manipulation_experiment/"  
echo "- Residual interruption: plots/residual_interruption_experiment/"
echo
echo "=== END PHASE 1 EXPERIMENTS ==="