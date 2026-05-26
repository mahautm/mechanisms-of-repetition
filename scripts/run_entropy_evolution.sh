#!/bin/bash
#SBATCH --job-name=entropy_evolution
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/entropy_evolution_%j.out
#SBATCH --error=logs/entropy_evolution_%j.err

# Run entropy evolution analysis for both Pythia-70m and Pythia-1.4b

cd /home/mmahaut/projects/parrots
mkdir -p logs entropy_evolution_results

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0

echo "================================================"
echo "Entropy Evolution Analysis"
echo "================================================"
date

# Run for Pythia-70m
echo ""
echo "🔬 Running Pythia-70m entropy evolution..."
python entropy_evolution_analysis.py \
    --model_name EleutherAI/pythia-70m \
    --checkpoints step1 step1000 step5000 step10000 step100000 steplatest \
    --n_samples 200 \
    --batch_size 16 \
    --output_dir entropy_evolution_results

# Run for Pythia-1.4b  
echo ""
echo "🔬 Running Pythia-1.4b entropy evolution..."
python entropy_evolution_analysis.py \
    --model_name EleutherAI/pythia-1.4b \
    --checkpoints step1 step1000 step5000 step10000 step100000 steplatest \
    --n_samples 200 \
    --batch_size 8 \
    --output_dir entropy_evolution_results

echo ""
echo "✅ All entropy evolution analysis complete!"
date
