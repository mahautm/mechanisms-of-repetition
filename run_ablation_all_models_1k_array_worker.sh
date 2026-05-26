#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=abl_1k
#SBATCH --output=logs/ablation_1k_%A_%a.out
#SBATCH --error=logs/ablation_1k_%A_%a.err

set -euo pipefail

ROOT="/home/mmahaut/projects/parrots"
TASK_ID="${SLURM_ARRAY_TASK_ID}"

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${CHECKPOINTS_CSV:?CHECKPOINTS_CSV is required}"
: "${N_SAMPLES:?N_SAMPLES is required}"
: "${RAW_POOL_SIZE:?RAW_POOL_SIZE is required}"
: "${RAW_PROMPT_TOKENS:?RAW_PROMPT_TOKENS is required}"
: "${PROMPT_GENERATION_TOKENS:?PROMPT_GENERATION_TOKENS is required}"
: "${BATCH_SIZE:?BATCH_SIZE is required}"
: "${MAX_PROMPT_TOKENS:?MAX_PROMPT_TOKENS is required}"
: "${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
: "${CHUNK_COUNT:?CHUNK_COUNT is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${USE_BNB:?USE_BNB is required}"
: "${LAYER_COUNT:?LAYER_COUNT is required}"

mkdir -p "$OUT_DIR" "$ROOT/logs"

TASKS_PER_CKPT=$((LAYER_COUNT * CHUNK_COUNT))
CKPT_IDX=$((TASK_ID / TASKS_PER_CKPT))
REMAINDER=$((TASK_ID % TASKS_PER_CKPT))
IFS='|' read -r -a CHECKPOINTS <<< "$CHECKPOINTS_CSV"
CKPT_COUNT="${#CHECKPOINTS[@]}"

LAYER=$((REMAINDER / CHUNK_COUNT))
CHUNK_INDEX=$((REMAINDER % CHUNK_COUNT))

if (( CKPT_IDX < 0 || CKPT_IDX >= CKPT_COUNT )); then
  echo "Invalid checkpoint index: $CKPT_IDX for task $TASK_ID" >&2
  exit 2
fi

CHECKPOINT="${CHECKPOINTS[$CKPT_IDX]}"
CKPT_SAFE="${CHECKPOINT//\//_}"
CKPT_SAFE="${CKPT_SAFE//:/_}"
TASK_OUT_DIR="$OUT_DIR/$CKPT_SAFE"
TASK_OUT_DIR="$TASK_OUT_DIR/chunk_${CHUNK_INDEX}_of_${CHUNK_COUNT}"
mkdir -p "$TASK_OUT_DIR"

PY_CMD=(
  python -u "$ROOT/ablation_head_cycle_icl_natural.py"
  --model_name "$MODEL_NAME"
  --checkpoints
  "$CHECKPOINT"
)
PY_CMD+=(
  --layer "$LAYER"
  --n_samples "$N_SAMPLES"
  --raw_pool_size "$RAW_POOL_SIZE"
  --raw_prompt_tokens "$RAW_PROMPT_TOKENS"
  --prompt_generation_tokens "$PROMPT_GENERATION_TOKENS"
  --batch_size "$BATCH_SIZE"
  --max_prompt_tokens "$MAX_PROMPT_TOKENS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --chunk_index "$CHUNK_INDEX"
  --chunk_count "$CHUNK_COUNT"
  --output_dir "$TASK_OUT_DIR"
)

if [[ "$USE_BNB" == "1" ]]; then
  PY_CMD+=(--use_bnb)
fi

CMD_STR="$(printf '%q ' "${PY_CMD[@]}")"

echo "[ablation-1k] model=$MODEL_NAME task=$TASK_ID layer=$LAYER checkpoint=$CHECKPOINT"
echo "[ablation-1k] n_samples=$N_SAMPLES batch=$BATCH_SIZE chunk=$CHUNK_INDEX/$CHUNK_COUNT out=$TASK_OUT_DIR"

srun --partition=alien --qos=alien --exclude=node044 --gres=gpu:1 --mem=96G \
  bash -lc "source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && $CMD_STR"
