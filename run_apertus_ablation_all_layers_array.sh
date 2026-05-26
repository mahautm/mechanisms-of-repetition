#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=abl_apertus_all
#SBATCH --output=logs/ablation_apertus_all_layers_%A_%a.out
#SBATCH --error=logs/ablation_apertus_all_layers_%A_%a.err
#SBATCH --array=0-31%4

set -euo pipefail

ROOT="/home/mmahaut/projects/parrots"
LAYER="${SLURM_ARRAY_TASK_ID}"
N_SAMPLES="${N_SAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs_ablation_head_cycle_apertus_all_layers}"

mkdir -p "$OUT_DIR" "$ROOT/logs"

echo "[Apertus ablation] layer=$LAYER n_samples=$N_SAMPLES"

srun --partition=alien --qos=alien --exclude=node044 --gres=gpu:1 --mem=96G \
  /bin/bash -c "source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && python -u '$ROOT/ablation_head_cycle_icl_natural.py' \
    --model_name 'swiss-ai/Apertus-8B-2509' \
    --checkpoints steplatest \
    --layer '$LAYER' \
    --n_samples '$N_SAMPLES' \
    --batch_size '$BATCH_SIZE' \
    --max_prompt_tokens '$MAX_PROMPT_TOKENS' \
    --max_new_tokens '$MAX_NEW_TOKENS' \
    --output_dir '$OUT_DIR' \
    --use_bnb"
