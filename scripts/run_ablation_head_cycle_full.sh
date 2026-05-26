#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-EleutherAI/pythia-1.4b}"
LAYER="${2:-19}"

echo "Running full ablation sweep: model=${MODEL_NAME} layer=${LAYER}"

srun \
  --partition=alien \
  --qos=alien \
  --exclude=node044 \
  --gres=gpu:1 \
  --mem=80G \
  --time=06:00:00 \
  python /home/mmahaut/projects/parrots/ablation_head_cycle_evolution.py \
    --model_name "${MODEL_NAME}" \
    --checkpoints step1 step1000 step5000 step10000 step100000 steplatest \
    --layer "${LAYER}" \
    --n_samples 120 \
    --batch_size 8 \
    --max_prompt_tokens 64 \
    --max_new_tokens 128 \
    --do_sample \
    --temperature 0.8 \
    --top_p 0.9 \
    --cycle_sizes 2,3,4,5,6 \
    --output_dir /home/mmahaut/projects/parrots/outputs_ablation_head_cycle

echo "Full run complete."
