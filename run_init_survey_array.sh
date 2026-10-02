#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=init_survey
#SBATCH --output=logs/init_survey_%A_%a.out
#SBATCH --error=logs/init_survey_%A_%a.err
# Baseline-only repetition survey over independent initialisations (reviewer: provenance and
# number/architecture of initialisations). One task per "model|checkpoint" line of
# init_survey_models.txt. Uses the paired, length-matched prompt set (same prompts everywhere).
set -euo pipefail
ROOT="/home/mmahaut/projects/parrots"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/init_survey_models.txt")
MODEL="${LINE%%|*}"
CKPT="${LINE##*|}"
OUT="$ROOT/outputs_init_survey/${MODEL//\//_}/$CKPT"
mkdir -p "$OUT"
echo "[init-survey] model=$MODEL checkpoint=$CKPT out=$OUT"
"/home/mmahaut/.conda/envs/parr/bin/python" -u "$ROOT/ablation_head_cycle_icl_natural.py" \
  --model_name "$MODEL" --checkpoints "$CKPT" --layer 0 --baseline_only \
  --prompt_mode paired --n_samples "${N_SAMPLES:-1000}" --raw_pool_size "${RAW_POOL_SIZE:-1500}" \
  --raw_prompt_tokens 32 --max_new_tokens 1000 --batch_size "${BATCH_SIZE:-64}" \
  --output_dir "$OUT"
