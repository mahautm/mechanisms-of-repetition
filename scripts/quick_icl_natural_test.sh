#!/usr/bin/env bash
# Quick inline ICL+Natural validation (no external test folder)

set -e
cd /home/mmahaut/projects/parrots

echo "Quick ICL+Natural validation test"
echo "===================================="

# Run on Pythia-70m L0, steplatest only, minimal samples
srun \
  --partition=alien \
  --qos=alien \
  --exclude=node044 \
  --mem=24G \
  --gres=gpu:1 \
  --time=20:00 \
    bash -lc 'source ~/.bashrc && conda activate parr && python ablation_head_cycle_icl_natural.py \
    --model_name EleutherAI/pythia-70m \
    --layer 0 \
    --heads "0,1" \
    --n_samples 20 \
    --checkpoints steplatest \
        --output_dir ./outputs_ablation_head_cycle'

# Check output
echo ""
echo "Checking for output CSV..."
if [ -f "./outputs_ablation_head_cycle/head_cycle_ablation_EleutherAI_pythia-70m_L0_icl_natural.csv" ]; then
    echo "✓ Output found!"
    wc -l ./outputs_ablation_head_cycle/head_cycle_ablation_EleutherAI_pythia-70m_L0_icl_natural.csv
    head -3 ./outputs_ablation_head_cycle/head_cycle_ablation_EleutherAI_pythia-70m_L0_icl_natural.csv
else
    echo "✗ No output file found"
    exit 1
fi
