#!/bin/bash
#SBATCH --job-name=olmo_fallback_alluvial
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/olmo_attention_fallback_alluvial_%j.out
#SBATCH --error=logs/olmo_attention_fallback_alluvial_%j.err

# Parameters
MODEL_NAME="allenai/OLMo-1B-hf"
CHECKPOINT="step425000-tokens1781B"  # Best checkpoint: 33 No-Cycle-ICL out of 1000
N_SAMPLES=1000  # Like alluvial pipeline
SEED=${1:-42}
OUTPUT_DIR="./plots/attention_fallback_alluvial_allenai_OLMo-1B-hf_${CHECKPOINT}_seed${SEED}"
MAX_LENGTH=32  # Like alluvial
MAX_NEW_TOKENS=1000  # Like alluvial
N_CYCLES=4  # Like alluvial
BATCH_SIZE=64  # Faster generation

echo "=========================================="
echo "OLMo Attention Fallback (Alluvial-Style)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT"
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
    --revision "$CHECKPOINT" \
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
