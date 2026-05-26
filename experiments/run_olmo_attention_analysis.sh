#!/bin/bash
# Run OLMo attention analysis across all layers
# This replicates the Pythia checkpoint analysis but for a single OLMo model

MODEL="allenai/OLMo-1B-hf"
MAX_LAYERS=16
TARGET_LAYER=12  # Proportional to Pythia layer 19

echo "=========================================="
echo "OLMo Attention Analysis - Full Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Layers: $MAX_LAYERS"
echo "Target layer for comparison: $TARGET_LAYER"
echo "=========================================="
echo ""

# Option 1: Run single target layer (quick)
run_single_layer() {
    echo "Running single layer analysis (layer $TARGET_LAYER)..."
    sbatch experiments/slurm_olmo_attention.sh $TARGET_LAYER
}

# Option 2: Run all layers (comprehensive)
run_all_layers() {
    echo "Submitting jobs for all $MAX_LAYERS layers..."
    for layer in $(seq 0 $((MAX_LAYERS - 1))); do
        echo "  Submitting layer $layer..."
        sbatch experiments/slurm_olmo_attention.sh $layer
        sleep 1  # Avoid overwhelming scheduler
    done
}

# Option 3: Run key layers only (middle ground)
run_key_layers() {
    # Analyze layers at 25%, 50%, 75%, 100% depth
    KEY_LAYERS=(3 7 12 15)
    echo "Running key layers: ${KEY_LAYERS[@]}..."
    for layer in "${KEY_LAYERS[@]}"; do
        echo "  Submitting layer $layer..."
        sbatch experiments/slurm_olmo_attention.sh $layer
        sleep 1
    done
}

# Parse command line argument
case "${1:-single}" in
    single)
        run_single_layer
        ;;
    all)
        run_all_layers
        ;;
    key)
        run_key_layers
        ;;
    *)
        echo "Usage: $0 {single|all|key}"
        echo "  single - Run target layer $TARGET_LAYER only (default)"
        echo "  all    - Run all $MAX_LAYERS layers"
        echo "  key    - Run key layers (25%, 50%, 75%, 100% depth)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Jobs submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/olmo_attention_*.out"
echo "=========================================="
