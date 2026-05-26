#!/bin/bash
#SBATCH --job-name=olmo_sample
#SBATCH --output=logs/olmo_sample_%j.out
#SBATCH --error=logs/olmo_sample_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# OLMo Sample Experiment on SLURM
# ================================
# Run sample experiment with OLMo on sample data
#
# Usage:
#   sbatch experiments/slurm_olmo_sample.sh
#
# Or with custom parameters:
#   sbatch --export=MODEL=allenai/OLMo-7B-hf,SAMPLE_SIZE=100 experiments/slurm_olmo_sample.sh

set -e

# Configuration from environment or defaults
MODEL="${MODEL:-allenai/OLMo-1B-hf}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/olmo_sample_experiment}"
DATA_PATH="${DATA_PATH:-data/human_lama_parrots_list_v1.csv}"

# Project setup
PROJECT_ROOT="/home/mmahaut/projects/parrots"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs

# Print configuration
echo "=========================================="
echo "OLMo Sample Experiment on SLURM"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Sample size: $SAMPLE_SIZE"
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

# Run experiment
echo "Starting experiment..."
echo ""

python experiments/olmo_sample_experiment.py \
    --model "$MODEL" \
    --sample-size "$SAMPLE_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --data-path "$DATA_PATH" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment completed successfully!"
    echo "Results in: $OUTPUT_DIR"
    echo "Report: $OUTPUT_DIR/experiment_report.md"
else
    echo "✗ Experiment failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/olmo_sample_${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit $EXIT_CODE
