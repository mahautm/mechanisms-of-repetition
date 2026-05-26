#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/mmahaut/projects/parrots"
OUT_ROOT="${ROOT_DIR}/outputs/pile_mitigations"
LOG_DIR="${ROOT_DIR}/logs/pile_mitigations"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

N_SAMPLES="${N_SAMPLES:-512}"
PROMPT_SIZE="${PROMPT_SIZE:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SEED="${SEED:-42}"

COMMON_SRUN=(
  --partition=alien
  --qos=alien
  --exclude=node044
  --gres=gpu:1
  --mem=96G
)

MODELS=(
  "meta-llama/Llama-3.2-1B"
  "facebook/opt-1.3b"
  "allenai/OLMo-1B-hf"
  "EleutherAI/pythia-1.4b"
)

TOP_P_VALUES=("none" "0.5" "0.9")

declare -a PIDS
declare -a TAGS

for model in "${MODELS[@]}"; do
  safe_model="$(echo "${model}" | tr '/' '_')"
  for p in "${TOP_P_VALUES[@]}"; do
    tag="${safe_model}_p${p}"
    log_file="${LOG_DIR}/${tag}.log"
    out_dir="${OUT_ROOT}/${safe_model}_p${p}"
    mkdir -p "${out_dir}"

    if [[ "${p}" == "none" ]]; then
      top_p_arg=""
    else
      top_p_arg="--top_p ${p}"
    fi

    echo "[LAUNCH] ${tag}"
    srun "${COMMON_SRUN[@]}" --time=08:00:00 \
      bash -lc "source ~/.bashrc && conda activate parr && python '${ROOT_DIR}/run_pile_top_p_natural_icl.py' \
        --model_name '${model}' \
        --output_dir '${out_dir}' \
        --n_samples '${N_SAMPLES}' \
        --prompt_size '${PROMPT_SIZE}' \
        --max_new_tokens '${MAX_NEW_TOKENS}' \
        --seed '${SEED}' ${top_p_arg}" > "${log_file}" 2>&1 &

    PIDS+=("$!")
    TAGS+=("${tag}")
  done
done

echo "Submitted ${#PIDS[@]} jobs. Waiting for completion..."

fail_count=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[DONE] ${TAGS[$i]}"
  else
    echo "[FAIL] ${TAGS[$i]}"
    fail_count=$((fail_count + 1))
  fi
done

echo "All jobs complete. Failures: ${fail_count}"
exit "${fail_count}"
