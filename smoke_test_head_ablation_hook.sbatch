#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=hp_smoke
#SBATCH --array=0-2
#SBATCH --output=logs/head_policy_smoke_%A_%a.out
#SBATCH --error=logs/head_policy_smoke_%A_%a.err

set -euo pipefail

ROOT="/home/mmahaut/projects/parrots"
MODEL="EleutherAI/pythia-70m"
POLICIES=(
  "h1_selective_balanced|$ROOT/outputs/head_policy_library/pythia70m/policy_lambda1.0_k2_factor1.25.json"
  "h2_selective_conservative|$ROOT/outputs/head_policy_library/pythia70m/policy_lambda2.0_k2_factor1.10.json"
  "h3_selective_aggressive|$ROOT/outputs/head_policy_library/pythia70m/policy_lambda0.5_k4_factor1.50.json"
)

IFS='|' read -r HYP POLICY <<< "${POLICIES[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$ROOT/outputs/head_policy_eval/smoke_fix_hook/${HYP}/pythia70m"
mkdir -p "$OUT_DIR"

echo "[SMOKE:$SLURM_ARRAY_TASK_ID] $MODEL | $HYP | policy=$POLICY"
PYTHON="/home/mmahaut/.conda/envs/parr/bin/python"
"$PYTHON" "$ROOT/run_head_intervention_forced_cycles.py" \
  --model_name "$MODEL" \
  --policy_json "$POLICY" \
  --n_samples 48 \
  --batch_size 2 \
  --max_prompt_tokens 256 \
  --max_new_tokens 96 \
  --delta_nat_threshold 0.20 \
  --epsilon_icl 0.02 \
  --output_dir "$OUT_DIR"
