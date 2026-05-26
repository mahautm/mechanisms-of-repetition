#!/bin/bash
#SBATCH --job-name=ent_3way_70m
#SBATCH --output=outputs_entropy_3way/logs/entropy_3way_70m.out
#SBATCH --error=outputs_entropy_3way/logs/entropy_3way_70m.err
#SBATCH --mem=64G
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0
cd ~/projects/parrots
mkdir -p outputs_entropy_3way/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running 3-way entropy analysis for Pythia-70m with 500 samples"
poetry run python entropy_3way_categorization.py \
    --model-name=EleutherAI/pythia-70m \
    --n-samples=500 \
    --batch-size=8 \
    --checkpoints="step1,step1000,step5000,step10000,step100000,steplatest"

echo "Done!"
