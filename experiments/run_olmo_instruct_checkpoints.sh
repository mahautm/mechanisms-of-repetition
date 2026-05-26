#!/bin/bash
# Run OLMo-2 Instruct attention analysis across SFT checkpoints
# This analyzes how attention patterns evolve during instruction tuning

TARGET_LAYER=${1:-12}  # Default to layer 12 (proportional to Pythia layer 19)

# OLMo-1B-0724-hf checkpoints - full training from pretraining through SFT
# Selected 6 checkpoints evenly distributed across 1446 total checkpoints
CHECKPOINTS=(
    "step0-tokens0B"            # Start of training (0%)
    "step288000-tokens603B"     # ~20% training
    "step577000-tokens1209B"    # ~40% training
    "step865000-tokens1813B"    # ~60% training
    "step1165000-tokens2442B"   # ~80% training
    "step1454000-tokens3048B"   # Final (100%)
)

echo "=========================================="
echo "OLMo-1B-0724 Full Evolution Analysis"
echo "=========================================="
echo "Model: allenai/OLMo-1B-0724-hf"
echo "Target layer: $TARGET_LAYER"
echo "Checkpoints: ${#CHECKPOINTS[@]} (SFT training)"
echo "=========================================="
echo ""

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "Submitting job for checkpoint: $checkpoint"
    sbatch experiments/slurm_olmo_instruct_attention.sh $TARGET_LAYER $checkpoint
    sleep 1  # Avoid overwhelming scheduler
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Monitor with:"
echo "  squeue -u \$USER | grep olmo_inst"
echo "  tail -f logs/olmo_instruct_*.out"
echo ""
echo "Results will be in:"
echo "  outputs/olmo_attention/allenai/OLMo-1B-0724-hf/{checkpoint}/layer_${TARGET_LAYER}/"
echo "=========================================="
