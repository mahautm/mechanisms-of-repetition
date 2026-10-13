#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044,node041
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=tf_abl
#SBATCH --output=logs/tf_%x_%A_%a.out
#SBATCH --error=logs/tf_%x_%A_%a.err
# Teacher-forced ablation worker. STAGE=baseline|group: one array task per checkpoint.
# STAGE=single: one task per (checkpoint, block of LAYERS_PER_TASK layers).

set -euo pipefail
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
: "${STAGE:?}" "${MODEL_NAME:?}" "${CHECKPOINTS_CSV:?}" "${OUT_DIR:?}"
ABLATION="${ABLATION:-zero}"
IFS='|' read -r -a CKPTS <<< "$CHECKPOINTS_CSV"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA=()
if [[ "$STAGE" == "single" ]]; then
  : "${LAYER_COUNT:?}"
  PER="${LAYERS_PER_TASK:-2}"
  BLOCKS=$(( (LAYER_COUNT + PER - 1) / PER ))
  CKPT="${CKPTS[$((TASK / BLOCKS))]}"
  FIRST=$(( (TASK % BLOCKS) * PER ))
  LAST=$(( FIRST + PER - 1 < LAYER_COUNT - 1 ? FIRST + PER - 1 : LAYER_COUNT - 1 ))
  EXTRA=(--layers $(seq "$FIRST" "$LAST"))
else
  CKPT="${CKPTS[$TASK]}"
fi
echo "[tf] stage=$STAGE model=$MODEL_NAME ckpt=$CKPT ablation=$ABLATION ${EXTRA[*]:-}"
set +u; source ~/.bashrc
conda activate parr; set -u
python -u teacher_forced_ablation.py --stage "$STAGE" --model_name "$MODEL_NAME" --checkpoint "$CKPT" \
  --output_dir "$OUT_DIR" --ablation "$ABLATION" "${EXTRA[@]}" ${EXTRA_ARGS:-}
