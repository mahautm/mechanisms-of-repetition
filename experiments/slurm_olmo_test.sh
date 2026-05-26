#!/bin/bash
#SBATCH --job-name=olmo_test
#SBATCH --output=logs/olmo_test_%j.out
#SBATCH --error=logs/olmo_test_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Quick OLMo Test on SLURM
# =========================
# Fast test with small sample to verify setup
#
# Usage:
#   sbatch experiments/slurm_olmo_test.sh

set -e

# Configuration
MODEL="allenai/OLMo-1B-hf"
SAMPLE_SIZE=10
OUTPUT_DIR="outputs/olmo_quick_test"

# Project setup
PROJECT_ROOT="/home/mmahaut/projects/parrots"
cd "$PROJECT_ROOT"

mkdir -p logs

echo "=========================================="
echo "Quick OLMo Test on SLURM"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Sample size: $SAMPLE_SIZE (quick test)"
echo "=========================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate parr

# Check GPU
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

# Run quick test
echo "Running quick test..."
python experiments/olmo_sample_experiment.py \
    --model "$MODEL" \
    --sample-size "$SAMPLE_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test PASSED!"
    echo "You can now run larger experiments."
else
    echo "✗ Test FAILED!"
    echo "Fix issues before running larger experiments."
fi

exit $EXIT_CODE
