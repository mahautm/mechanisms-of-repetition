#!/bin/bash
#SBATCH --job-name=cycle_evolution_orig_learnt
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cycle_evolution_orig_learnt_%j.out
#SBATCH --error=logs/cycle_evolution_orig_learnt_%j.err

# Cycle Evolution: Original vs Learnt Repetition
# Runs for both Pythia-70m and Pythia-1.4b

cd /home/mmahaut/projects/parrots
mkdir -p logs cycle_evolution_results

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0

echo "================================================"
echo "Cycle Evolution: Original vs Learnt Repetition"
echo "================================================"
date

# Run for Pythia-70m
echo ""
echo "🔬 Running Pythia-70m cycle evolution..."
python cycle_evolution_original_vs_learnt.py \
    --model_name EleutherAI/pythia-70m \
    --checkpoints step1 step1000 step5000 step10000 step100000 steplatest \
    --n_samples 300 \
    --batch_size 16 \
    --output_dir cycle_evolution_results

# Run for Pythia-1.4b
echo ""
echo "🔬 Running Pythia-1.4b cycle evolution..."
python cycle_evolution_original_vs_learnt.py \
    --model_name EleutherAI/pythia-1.4b \
    --checkpoints step1 step1000 step5000 step10000 step100000 steplatest \
    --n_samples 300 \
    --batch_size 8 \
    --output_dir cycle_evolution_results

echo ""
echo "✅ Cycle evolution analysis complete!"
date
