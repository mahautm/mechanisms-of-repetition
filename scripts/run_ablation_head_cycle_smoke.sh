#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-EleutherAI/pythia-70m}"
LAYER="${2:-4}"
CHECKPOINTS="${3:-step1 steplatest}"

echo "Running smoke ablation: model=${MODEL_NAME} layer=${LAYER} checkpoints=${CHECKPOINTS}"

srun \
  --partition=alien \
  --qos=alien \
  --exclude=node044 \
  --gres=gpu:1 \
  --mem=32G \
  --time=00:30:00 \
  python /home/mmahaut/projects/parrots/ablation_head_cycle_evolution.py \
    --model_name "${MODEL_NAME}" \
    --checkpoints ${CHECKPOINTS} \
    --layer "${LAYER}" \
    --heads "0,1,2,3" \
    --n_samples 24 \
    --batch_size 6 \
    --max_prompt_tokens 48 \
    --max_new_tokens 96 \
    --do_sample \
    --temperature 0.8 \
    --top_p 0.9 \
    --output_dir /home/mmahaut/projects/parrots/outputs_ablation_head_cycle

echo "Smoke run complete."
