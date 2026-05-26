#!/bin/bash
#SBATCH --job-name=olmo_attn
#SBATCH --output=logs/olmo_attention_%j.out
#SBATCH --error=logs/olmo_attention_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# OLMo Attention Analysis on SLURM
# =================================
# Run attention contrast analysis on OLMo model
# Based on Pythia aa_fortu experiments
#
# Usage:
#   sbatch experiments/slurm_olmo_attention.sh [LAYER]

set -e

# Configuration from arguments or defaults
LAYER=${1:-12}  # Default to layer 12 (≈75% depth, similar to Pythia layer 19)
CHECKPOINT=${2:-""}  # Optional: specific checkpoint revision (e.g., step100000-tokens419B)
MODEL_NAME="allenai/OLMo-1B-hf"  # Use OLMo-1B-hf for Python 3.9 compatibility
MAX_LAYERS=16
N_CYCLES=4
MAX_LENGTH=32
BATCH_SIZE=8
N_SAMPLES=1000
MAX_NEW_TOKENS=1000

# Output directory - include checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    OUTPUT_DIR="outputs/olmo_attention/${MODEL_NAME}/${CHECKPOINT}/layer_${LAYER}"
    MODEL_REVISION="--revision=${CHECKPOINT}"
else
    OUTPUT_DIR="outputs/olmo_attention/${MODEL_NAME}/main/layer_${LAYER}"
    MODEL_REVISION=""
fi
LENS_PATH="lenses_multihead/OLMo_1B"  # For future lens training

# Project setup
PROJECT_ROOT="/home/mmahaut/projects/parrots"
cd "$PROJECT_ROOT"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Print configuration
echo "=========================================="
echo "OLMo Attention Analysis on SLURM"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_NAME"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
else
    echo "Checkpoint: main (final)"
fi
echo "Target layer: $LAYER / $MAX_LAYERS"
echo "N cycles: $N_CYCLES"
echo "Max length: $MAX_LENGTH"
echo "Samples: $N_SAMPLES"
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

# Run attention analysis using OLMo-specific pipeline
echo "Starting attention analysis..."
echo ""

python experiments/olmo_attention_pipeline.py \
    --model-name="$MODEL_NAME" \
    $MODEL_REVISION \
    --single-lens="$LAYER" \
    --max-layer-idx="$MAX_LAYERS" \
    --n-cycles="$N_CYCLES" \
    --max-length="$MAX_LENGTH" \
    --max-new-tokens="$MAX_NEW_TOKENS" \
    --batch-size="$BATCH_SIZE" \
    --n-samples="$N_SAMPLES" \
    --no-head-analysis

EXIT_CODE=$?

# Move outputs to layer-specific directory
mv *.png "$OUTPUT_DIR/" 2>/dev/null || true
mv *.npy "$OUTPUT_DIR/" 2>/dev/null || true

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Analysis completed successfully!"
    echo "Results in: $OUTPUT_DIR"
    echo ""
    echo "Expected outputs:"
    echo "  - Natural repetition heatmap"
    echo "  - ICL repetition heatmap"
    echo "  - No-cycle ICL heatmap"
    echo "  - Data and repetition indices"
else
    echo "✗ Analysis failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/olmo_attention_${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit $EXIT_CODE
