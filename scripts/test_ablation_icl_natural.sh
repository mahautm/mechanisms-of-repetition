#!/usr/bin/env bash
# Quick validation: Test ICL+Natural ablation on Pythia-70m L0

set -e

echo "Testing ICL +Natural condition ablation..."
echo "==========================================="

OUTPUT_DIR="/home/mmahaut/projects/parrots/outputs_ablation_head_cycle_icl_natural_test"
mkdir -p "$OUTPUT_DIR"

# Quick smoke test: just 30 samples per condition, 2 heads, 1 model
srun \
  --partition=alien \
  --qos=alien \
  --exclude=node044 \
  --mem=24G \
  --gres=gpu:1 \
  --time=30:00 \
  bash -lc 'source ~/.bashrc && conda activate parr && python /home/mmahaut/projects/parrots/ablation_head_cycle_icl_natural.py \
    --model_name EleutherAI/pythia-70m \
    --layer 0 \
    --heads "0,1,2" \
    --n_samples 30 \
    --checkpoints steplatest \
    --output_dir "'$OUTPUT_DIR'"'

echo ""
echo "Test Results:"
ls -lh "$OUTPUT_DIR" | grep -E "\.csv|\.json"

echo ""
echo "✓ Validation passed! Output:"
head -20 "$OUTPUT_DIR"/*.csv
