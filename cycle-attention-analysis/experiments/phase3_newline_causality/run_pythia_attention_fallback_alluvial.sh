#!/bin/bash
#SBATCH --job-name=pythia_fallback_alluvial
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pythia_attention_fallback_alluvial_%j.out
#SBATCH --error=logs/pythia_attention_fallback_alluvial_%j.err

# Parameters
MODEL_NAME="EleutherAI/pythia-1.4b"
N_SAMPLES=1000  # Like alluvial pipeline
SEED=${1:-42}
OUTPUT_DIR="./plots/attention_fallback_alluvial_EleutherAI_pythia-1.4b_seed${SEED}"
MAX_LENGTH=32  # Like alluvial
MAX_NEW_TOKENS=1000  # Like alluvial
N_CYCLES=4  # Like alluvial
BATCH_SIZE=64  # Faster generation
TARGET_LAYER=19  # 19/24 layers ≈ 75% depth

echo "=========================================="
echo "Pythia-1.4b Attention Fallback (Alluvial-Style)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_NAME"
echo "Target layer: $TARGET_LAYER"
echo "Samples: $N_SAMPLES"
echo "Seed: $SEED"
echo "Max length: $MAX_LENGTH"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "N cycles: $N_CYCLES"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Activate environment
source ~/.bashrc
conda activate parr

# Run analysis
python compare_attention_fallback_alluvial_style.py \
    --model_name "$MODEL_NAME" \
    --target_layer $TARGET_LAYER \
    --n_samples $N_SAMPLES \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --max_length $MAX_LENGTH \
    --max_new_tokens $MAX_NEW_TOKENS \
    --n_cycles $N_CYCLES \
    --batch_size $BATCH_SIZE

echo ""
echo "=========================================="
echo "✓ Analysis completed!"
echo "Results in: $OUTPUT_DIR"
echo "=========================================="
