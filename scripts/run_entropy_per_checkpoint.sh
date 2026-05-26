#!/bin/bash
#SBATCH --job-name=entropy_ckpts
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mmahaut/projects/parrots/logs/entropy_checkpoints_%j.out
#SBATCH --error=/home/mmahaut/projects/parrots/logs/entropy_checkpoints_%j.err

# Logit entropy evolution analysis across all checkpoints
# Creates plots for each checkpoint showing entropy vs cycle number

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0

cd /home/mmahaut/projects/parrots
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_SAMPLES=200
OUTPUT_DIR="./outputs_entropy_evolution"

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=========================================="
echo "Logit Entropy Evolution per Checkpoint"
echo "=========================================="

# Pythia-1.4b
echo ""
echo "Processing Pythia-1.4b..."
python3 logit_entropy_per_checkpoint.py \
    --model-name="EleutherAI/pythia-1.4b" \
    --n-samples=${N_SAMPLES} \
    --output-dir="${OUTPUT_DIR}"

# Pythia-70m
echo ""
echo "Processing Pythia-70m..."
python3 logit_entropy_per_checkpoint.py \
    --model-name="EleutherAI/pythia-70m" \
    --n-samples=${N_SAMPLES} \
    --output-dir="${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "All entropy analyses complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
