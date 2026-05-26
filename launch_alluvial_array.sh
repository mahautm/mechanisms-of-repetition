#!/usr/bin/env bash
#SBATCH --job-name=parrots-alluvial
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-95%40  # 12 conditions * 8 ranks = 96 jobs
#SBATCH --output=/home/mmahaut/projects/parrots/logs/pile_mitigations/array_%A_%a.out

source ~/.bashrc
conda activate parr

ROOT_DIR="/home/mmahaut/projects/parrots"
OUT_ROOT="${ROOT_DIR}/outputs/pile_mitigations"

N_SAMPLES=512
PROMPT_SIZE=512
MAX_NEW_TOKENS=1000
SEED=42
N_RANKS=8

MODELS=(
  "meta-llama/Llama-3.2-1B"
  "facebook/opt-1.3b"
  "allenai/OLMo-1B-hf"
  "EleutherAI/pythia-1.4b"
)
TOP_P_VALUES=("none" "0.5" "0.9")

# Map SLURM_ARRAY_TASK_ID to conditions
MODELS_P_LEN=${#TOP_P_VALUES[@]}
TOTAL_CONDITIONS=$(( ${#MODELS[@]} * MODELS_P_LEN ))

CONDITION_IDX=$(( SLURM_ARRAY_TASK_ID / N_RANKS ))
RANK_IDX=$(( SLURM_ARRAY_TASK_ID % N_RANKS ))

MODEL_IDX=$(( CONDITION_IDX / MODELS_P_LEN ))
TOP_P_IDX=$(( CONDITION_IDX % MODELS_P_LEN ))

MODEL="${MODELS[$MODEL_IDX]}"
TOP_P="${TOP_P_VALUES[$TOP_P_IDX]}"

SAFE_MODEL="$(echo "${MODEL}" | tr '/' '_')"
TAG="${SAFE_MODEL}_p${TOP_P}"
OUT_DIR="${OUT_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

if [[ "${TOP_P}" == "none" ]]; then
  TOP_P_ARG=""
else
  TOP_P_ARG="--top_p ${TOP_P}"
fi

echo "[START] Task $SLURM_ARRAY_TASK_ID -> Model=$MODEL, TopP=$TOP_P, Rank=$RANK_IDX"

python "${ROOT_DIR}/run_pile_top_p_natural_icl.py" \
    --model_name "${MODEL}" \
    --output_dir "${OUT_DIR}" \
    --n_samples "${N_SAMPLES}" \
    --prompt_size "${PROMPT_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    ${TOP_P_ARG} \
    --rank "${RANK_IDX}" \
    --n_ranks "${N_RANKS}"

echo "[DONE] Task $SLURM_ARRAY_TASK_ID"
