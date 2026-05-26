#!/bin/bash
# Launch OLMo attention fallback analysis with 10 different seeds

echo "🚀 Launching OLMo attention fallback with 10 different seeds"
echo "============================================================"

SEEDS=(42 123 456 789 1024 2048 3141 5926 8192 16384)

for seed in "${SEEDS[@]}"; do
    echo "Submitting job with seed: $seed"
    sbatch run_olmo_attention_fallback.sh $seed
    sleep 1  # Brief pause to avoid overwhelming scheduler
done

echo ""
echo "✅ All 10 jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch 'squeue -u \$USER'"
echo ""
echo "Results will be in:"
echo "  plots/attention_fallback_comparison_allenai_OLMo-1B-hf_seed*/"
