#!/usr/bin/env bash
#SBATCH --job-name=parrots-checkpoints
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-63%40  # 8 checkpoints * 8 ranks = 64 jobs
#SBATCH --output=/home/mmahaut/projects/parrots/logs/pile_mitigations/chkpt_%A_%a.out

source ~/.bashrc
conda activate parr

ROOT_DIR="/home/mmahaut/projects/parrots"
OUT_ROOT="${ROOT_DIR}/outputs/pile_checkpoints"

N_SAMPLES=512
PROMPT_SIZE=512
MAX_NEW_TOKENS=1000
SEED=42
N_RANKS=8

CHECKPOINTS=(
  "step1"
  "step1000"
  "step5000"
  "step10000"
  "step50000"
  "step100000"
  "step140000"
  "step143000"
)

# Map SLURM_ARRAY_TASK_ID to conditions
CHKPT_IDX=$(( SLURM_ARRAY_TASK_ID / N_RANKS ))
RANK_IDX=$(( SLURM_ARRAY_TASK_ID % N_RANKS ))

CHKPT="${CHECKPOINTS[$CHKPT_IDX]}"

MODEL="EleutherAI/pythia-1.4b"
SAFE_MODEL="EleutherAI_pythia-1.4b"
TAG="${SAFE_MODEL}_${CHKPT}"
OUT_DIR="${OUT_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

echo "[START] Task $SLURM_ARRAY_TASK_ID -> Checkpoint=$CHKPT, Rank=$RANK_IDX"

python "${ROOT_DIR}/run_pile_top_p_natural_icl.py" \
    --model_name "${MODEL}" \
    --revision "${CHKPT}" \
    --output_dir "${OUT_DIR}" \
    --n_samples "${N_SAMPLES}" \
    --prompt_size "${PROMPT_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --rank "${RANK_IDX}" \
    --n_ranks "${N_RANKS}"

echo "[DONE] Task $SLURM_ARRAY_TASK_ID"
