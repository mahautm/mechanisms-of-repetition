#!/bin/bash
#SBATCH --job-name=apertus_attn
#SBATCH --output=logs/apertus_attention_%j.out
#SBATCH --error=logs/apertus_attention_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=8

# Apertus Attention Analysis on SLURM
# ====================================
# Run attention contrast analysis on Apertus model
# Based on OLMo/Pythia aa_fortu experiments
#
# Usage:
#   sbatch experiments/slurm_apertus_attention.sh [LAYER] [CHECKPOINT]

set -e

# Configuration from arguments or defaults
LAYER=${1:-24}  # Default to layer 24 (≈75% depth, similar to Pythia layer 19)
CHECKPOINT=${2:-""}  # Optional: specific checkpoint revision (e.g., step1000000-tokens4200B)
MODEL_NAME="swiss-ai/Apertus-8B-2509"
MAX_LAYERS=32
N_CYCLES=4
MAX_LENGTH=32
BATCH_SIZE=1
N_SAMPLES=1000
MAX_NEW_TOKENS=256

# Output directory - include checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    OUTPUT_DIR="outputs/apertus_attention/${MODEL_NAME}/${CHECKPOINT}/layer_${LAYER}"
    MODEL_REVISION="--revision=${CHECKPOINT}"
else
    OUTPUT_DIR="outputs/apertus_attention/${MODEL_NAME}/main/layer_${LAYER}"
    MODEL_REVISION=""
fi

# Project setup
PROJECT_ROOT="/home/mmahaut/projects/parrots"
cd "$PROJECT_ROOT"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Print configuration
echo "=========================================="
echo "Apertus Attention Analysis"
echo "=========================================="
echo "Model: $MODEL_NAME"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "Layer: $LAYER (of $MAX_LAYERS)"
echo "Cycles: $N_CYCLES"
echo "Max length: $MAX_LENGTH"
echo "Batch size: $BATCH_SIZE"
echo "Samples: $N_SAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Load environment
source ~/.bashrc
conda activate parr

# Verify transformers version
echo "Checking transformers version..."
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
echo ""

# Run analysis with no_head_analysis flag for fast index generation
echo "Starting analysis..."
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
    echo "  - Data indices (for alluvial plots)"
    echo "  - Repetition indices (for alluvial plots)"
    echo "  - No-cycle ICL indices (for alluvial plots)"
else
    echo "✗ Analysis failed with exit code $EXIT_CODE"
    echo "Check logs for details"
fi
echo "=========================================="

exit $EXIT_CODE
