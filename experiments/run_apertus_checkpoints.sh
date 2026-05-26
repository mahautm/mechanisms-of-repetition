#!/bin/bash
# Launch Apertus checkpoint analysis for alluvial plots
# Usage: ./experiments/run_apertus_checkpoints.sh [LAYER]

LAYER=${1:-24}  # Default to layer 24 (75% depth)

echo "=========================================="
echo "Launching Apertus Checkpoint Analysis"
echo "Layer: $LAYER"
echo "=========================================="
echo ""

# 6 checkpoints spanning the full 15T token training
CHECKPOINTS=(
    "step50000-tokens210B"      # 1.4% - early training
    "step500000-tokens2100B"    # 14% - emerging patterns
    "step1000000-tokens4200B"   # 28% - mid-training
    "step1700000-tokens7232B"   # 48% - mature
    "step2300000-tokens12272B"  # 82% - late training
    "step2627139-tokens15T"     # 100% - final model
)

JOB_IDS=()

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo "Submitting job for checkpoint: $CHECKPOINT"
    JOB_ID=$(sbatch experiments/slurm_apertus_attention.sh "$LAYER" "$CHECKPOINT" | awk '{print $4}')
    JOB_IDS+=($JOB_ID)
    echo "  → Job ID: $JOB_ID"
    sleep 1
done

echo ""
echo "=========================================="
echo "Submitted ${#JOB_IDS[@]} jobs"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with: squeue -palien -u $USER"
echo "Check logs: tail -f logs/apertus_attention_<JOB_ID>.out"
echo "=========================================="
