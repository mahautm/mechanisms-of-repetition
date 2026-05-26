#!/bin/bash
#SBATCH --job-name=ent_3way_1.4b
#SBATCH --output=outputs_entropy_3way/logs/entropy_3way_1.4b.out
#SBATCH --error=outputs_entropy_3way/logs/entropy_3way_1.4b.err
#SBATCH --mem=64G
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0
cd ~/projects/parrots
mkdir -p outputs_entropy_3way/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running 3-way entropy analysis for Pythia-1.4b with 500 samples"
poetry run python entropy_3way_categorization.py \
    --model-name=EleutherAI/pythia-1.4b \
    --n-samples=500 \
    --batch-size=4 \
    --checkpoints="step1,step1000,step5000,step10000,step100000,steplatest"

echo "Done!"
