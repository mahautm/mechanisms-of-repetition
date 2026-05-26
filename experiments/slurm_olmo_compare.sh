#!/bin/bash
#SBATCH --job-name=olmo_compare
#SBATCH --output=logs/olmo_compare_%j.out
#SBATCH --error=logs/olmo_compare_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# OLMo vs Pythia Comparison on SLURM
# ===================================
# Run comparison experiment between OLMo and Pythia
#
# Usage:
#   sbatch experiments/slurm_olmo_compare.sh
#
# Or with custom parameters:
#   sbatch --export=SAMPLE_SIZE=50,OLMO_MODEL=allenai/OLMo-7B-hf experiments/slurm_olmo_compare.sh

set -e

# Configuration from environment or defaults
SAMPLE_SIZE="${SAMPLE_SIZE:-30}"
OLMO_MODEL="${OLMO_MODEL:-allenai/OLMo-1B-hf}"
PYTHIA_MODEL="${PYTHIA_MODEL:-EleutherAI/pythia-1.4b}"
PYTHIA_CHECKPOINT="${PYTHIA_CHECKPOINT:-steplatest}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/olmo_pythia_comparison}"

# Project setup
PROJECT_ROOT="/home/mmahaut/projects/parrots"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs

# Print configuration
echo "=========================================="
echo "OLMo vs Pythia Comparison on SLURM"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Sample size: $SAMPLE_SIZE"
echo "OLMo model: $OLMO_MODEL"
echo "Pythia model: $PYTHIA_MODEL (checkpoint: $PYTHIA_CHECKPOINT)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate parr

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run comparison
echo "Starting comparison experiment..."
echo ""

python experiments/compare_olmo_pythia.py \
    --sample-size "$SAMPLE_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --olmo-model "$OLMO_MODEL" \
    --pythia-model "$PYTHIA_MODEL" \
    --pythia-checkpoint "$PYTHIA_CHECKPOINT"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Comparison completed successfully!"
    echo "Results in: $OUTPUT_DIR"
    echo "Report: $OUTPUT_DIR/comparison_report.md"
    echo "Plots: $OUTPUT_DIR/comparison_plots/"
else
    echo "✗ Comparison failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/olmo_compare_${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit $EXIT_CODE
