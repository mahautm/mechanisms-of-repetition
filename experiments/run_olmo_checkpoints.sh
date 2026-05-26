#!/bin/bash
# Run OLMo attention analysis across multiple checkpoints
# This replicates the Pythia checkpoint-by-checkpoint analysis

TARGET_LAYER=${1:-12}  # Default to layer 12 (proportional to Pythia layer 19)

# OLMo-1B-hf checkpoints spread across training (6 checkpoints like Pythia)
# Selected from 351 available checkpoints, evenly distributed
CHECKPOINTS=(
    "step1000-tokens4B"        # Very early training (0.1%)
    "step343000-tokens1438B"   # ~46% training  
    "step425000-tokens1781B"   # ~58% training
    "step509000-tokens2134B"   # ~69% training
    "step593000-tokens2486B"   # ~80% training
    "step738020-tokens3094B"   # Final (100%)
)

echo "=========================================="
echo "OLMo Checkpoint Evolution Analysis"
echo "=========================================="
echo "Target layer: $TARGET_LAYER"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "=========================================="
echo ""

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "Submitting job for checkpoint: $checkpoint"
    sbatch experiments/slurm_olmo_attention.sh $TARGET_LAYER $checkpoint
    sleep 1  # Avoid overwhelming scheduler
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/olmo_attention_*.out"
echo ""
echo "Results will be in:"
echo "  outputs/olmo_attention/allenai/OLMo-1B-hf/{checkpoint}/layer_${TARGET_LAYER}/"
echo "=========================================="
