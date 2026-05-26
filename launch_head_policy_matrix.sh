#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mmahaut/projects/parrots"
MATRIX_DIR="$ROOT/outputs/head_policy_matrix"
MATRIX_CSV="$MATRIX_DIR/matrix.csv"
mkdir -p "$MATRIX_DIR" "$ROOT/logs"

srun --partition=alien --qos=alien --exclude=node044 bash -lc "source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && python '$ROOT/prepare_head_policy_library.py' --output_root '$ROOT/outputs/head_policy_library'"

srun --partition=alien --qos=alien --exclude=node044 bash -lc "source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && python '$ROOT/build_head_policy_matrix.py' --output_csv '$MATRIX_CSV' --policy_root '$ROOT/outputs/head_policy_library' --n_samples 48 --max_new_tokens 96 --max_prompt_tokens 256"

N=$(( $(wc -l < "$MATRIX_CSV") - 1 ))
if [[ "$N" -le 0 ]]; then
  echo "No matrix rows found" >&2
  exit 1
fi
MAX=$((N - 1))

JOB_ID=$(sbatch --array=0-${MAX} "$ROOT/run_head_policy_array_worker.sh" "$MATRIX_CSV" | awk '{print $4}')
echo "Launched array job: $JOB_ID with $N tasks"
echo "Track: squeue -j $JOB_ID"
