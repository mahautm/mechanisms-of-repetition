#!/bin/bash
#SBATCH --job-name=olmo2_inst
#SBATCH --output=logs/olmo2_instruct_final_%j.out
#SBATCH --error=logs/olmo2_instruct_final_%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

# OLMoE-1B-7B Instruct Analysis
# ============================================
# Run attention analysis on final instruction-tuned OLMoE model

set -e

# Configuration
LAYER=${1:-12}  # Default to layer 12
MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
MAX_LAYERS=16
N_CYCLES=4
MAX_LENGTH=32
BATCH_SIZE=1
N_SAMPLES=1000
MAX_NEW_TOKENS=1000

OUTPUT_DIR="outputs/olmoe_instruct_final/${MODEL_NAME}/layer_${LAYER}"

# Project setup
PROJECT_ROOT="/home/mmahaut/projects/parrots"
cd "$PROJECT_ROOT"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Print configuration
echo "=========================================="
echo "OLMo-2-1B-Instruct Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_NAME"
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

# Run attention analysis
echo "Starting attention analysis..."
echo ""

python experiments/olmo_attention_pipeline.py \
    --model-name="$MODEL_NAME" \
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
    echo "  - Natural repetition indices"
    echo "  - ICL repetition indices"
    echo "  - No-cycle ICL indices"
else
    echo "✗ Analysis failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/olmoe_instruct_final_${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit $EXIT_CODE
