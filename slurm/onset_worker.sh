#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044,node041
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=onset
#SBATCH --output=logs/onset_%A_%a.out
#SBATCH --error=logs/onset_%A_%a.err
# Generation under group ablation (onset_ablation.py). One task per (checkpoint, block of head sets).
set -euo pipefail
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
: "${CHECKPOINTS_CSV:?}"
N_SETS="${N_SETS:-27}"
PER="${SETS_PER_TASK:-9}"
BLOCKS=$(( (N_SETS + PER - 1) / PER ))
IFS='|' read -r -a CKPTS <<< "$CHECKPOINTS_CSV"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CKPT="${CKPTS[$((TASK / BLOCKS))]}"
FIRST=$(( (TASK % BLOCKS) * PER ))
LAST=$(( FIRST + PER - 1 < N_SETS - 1 ? FIRST + PER - 1 : N_SETS - 1 ))
PYTHON="${PYTHON:-/home/mmahaut/.conda/envs/parr/bin/python}"
echo "[onset] ckpt=$CKPT sets=$FIRST..$LAST"
"$PYTHON" -u onset_ablation.py --checkpoint "$CKPT" --set_index $(seq "$FIRST" "$LAST") ${EXTRA_ARGS:-}
