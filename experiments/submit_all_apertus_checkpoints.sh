#!/bin/bash
# Submit Apertus attention analysis for all checkpoints
# Layer 24 (75% depth, 24/32)

LAYER=24
CHECKPOINTS=(
    "step50000-tokens210B"
    "step500000-tokens2100B"
    "step1000000-tokens4200B"
    "step1700000-tokens7232B"
    "step2300000-tokens12272B"
    "step2627139-tokens15T"
)

echo "🚀 Submitting Apertus-8B attention analysis jobs"
echo "================================================"
echo "Layer: $LAYER (75% depth)"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo ""

cd /home/mmahaut/projects/parrots

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo "📤 Submitting: $CHECKPOINT"
    JOB_ID=$(sbatch experiments/slurm_apertus_attention.sh $LAYER $CHECKPOINT | awk '{print $4}')
    echo "   Job ID: $JOB_ID"
done

echo ""
echo "✅ All jobs submitted!"
echo "Monitor with: squeue -u $USER"
