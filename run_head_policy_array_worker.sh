#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=hp_eval
#SBATCH --output=logs/head_policy_array_%A_%a.out
#SBATCH --error=logs/head_policy_array_%A_%a.err

set -euo pipefail

ROOT="/home/mmahaut/projects/parrots"
MATRIX_CSV="${1:-$ROOT/outputs/head_policy_matrix/matrix.csv}"
IDX="${SLURM_ARRAY_TASK_ID}"

if [[ ! -f "$MATRIX_CSV" ]]; then
  echo "Missing matrix: $MATRIX_CSV" >&2
  exit 1
fi

ROW=$(python - <<PY
import pandas as pd
m = pd.read_csv("$MATRIX_CSV")
i = int("$IDX")
if i < 0 or i >= len(m):
    raise SystemExit(2)
r = m.iloc[i]
print("|".join(str(r[c]) for c in [
    "family","size","model_name","hypothesis","policy_json","n_samples","batch_size","max_prompt_tokens","max_new_tokens","use_bnb","delta_nat_threshold","epsilon_icl"
]))
PY
)

IFS='|' read -r FAMILY SIZE MODEL HYP POLICY N_SAMPLES BATCH MAX_PROMPT MAX_NEW USE_BNB D_NAT EPS_ICL <<< "$ROW"

SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
OUT_DIR="$ROOT/outputs/head_policy_eval/matrix/${FAMILY}_${SIZE}/${HYP}/${SAFE_MODEL}"
mkdir -p "$OUT_DIR"

USE_BNB_FLAG=""
if [[ "$USE_BNB" == "1" ]]; then
  USE_BNB_FLAG="--use_bnb"
fi

echo "[ARRAY:$IDX] $MODEL | $HYP | policy=$POLICY"
srun --partition=alien --qos=alien --exclude=node044 --gres=gpu:1 --mem=96G \
  /bin/bash -c "source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && python '$ROOT/run_head_intervention_forced_cycles.py' \
    --model_name '$MODEL' \
    --policy_json '$POLICY' \
    --n_samples '$N_SAMPLES' \
    --batch_size '$BATCH' \
    --max_prompt_tokens '$MAX_PROMPT' \
    --max_new_tokens '$MAX_NEW' \
    --delta_nat_threshold '$D_NAT' \
    --epsilon_icl '$EPS_ICL' \
    --output_dir '$OUT_DIR' \
    $USE_BNB_FLAG"
