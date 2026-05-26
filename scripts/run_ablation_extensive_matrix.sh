#!/usr/bin/env bash
set -u

# Extensive head-by-head ablation sweep across relevant models/layers/checkpoints.
# Uses srun for every Python workload.

ROOT_DIR="/home/mmahaut/projects/parrots"
OUT_DIR="${ROOT_DIR}/outputs_ablation_head_cycle"
LOG_DIR="${ROOT_DIR}/logs/ablation_extensive"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

N_SAMPLES="${N_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SEED="${SEED:-42}"
DATASET="${DATASET:-JeanKaddour/minipile}"

COMMON_SRUN=(
  --partition=alien
  --qos=alien
  --exclude=node044
  --gres=gpu:1
)

run_case () {
  local model_name="$1"
  local layer="$2"
  local checkpoints="$3"
  local mem="$4"
  local time_budget="$5"

  local safe_model
  safe_model="$(echo "${model_name}" | tr '/' '_')"
  local tag="${safe_model}_L${layer}"
  local log_file="${LOG_DIR}/${tag}.log"

  echo "=================================================================" | tee -a "${log_file}"
  echo "[START] model=${model_name} layer=${layer} checkpoints=${checkpoints}" | tee -a "${log_file}"
  echo "=================================================================" | tee -a "${log_file}"

  srun "${COMMON_SRUN[@]}" --mem="${mem}" --time="${time_budget}" \
    bash -lc "source ~/.bashrc && conda activate parr && python '${ROOT_DIR}/ablation_head_cycle_evolution.py' \
      --model_name '${model_name}' \
      --checkpoints ${checkpoints} \
      --layer '${layer}' \
      --n_samples '${N_SAMPLES}' \
      --batch_size '${BATCH_SIZE}' \
      --max_prompt_tokens '${MAX_PROMPT_TOKENS}' \
      --max_new_tokens '${MAX_NEW_TOKENS}' \
      --seed '${SEED}' \
      --dataset '${DATASET}' \
      --do_sample \
      --temperature 0.8 \
      --top_p 0.9 \
      --cycle_sizes 2,3,4,5,6 \
      --output_dir '${OUT_DIR}'" >> "${log_file}" 2>&1

  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "[FAIL] ${tag} exit=${status}" | tee -a "${log_file}"
    return ${status}
  fi

  local overall_csv="${OUT_DIR}/head_cycle_ablation_${safe_model}_L${layer}_overall_delta.csv"
  local cycle_csv="${OUT_DIR}/head_cycle_ablation_${safe_model}_L${layer}_cycle_delta.csv"

  if [[ -f "${overall_csv}" && -f "${cycle_csv}" ]]; then
    srun --partition=alien --qos=alien --exclude=node044 --time=00:15:00 --mem=8G \
      bash -lc "source ~/.bashrc && conda activate parr && python '${ROOT_DIR}/analyze_ablation_head_cycle.py' \
        --overall_delta_csv '${overall_csv}' \
        --cycle_delta_csv '${cycle_csv}' \
        --output_dir '${OUT_DIR}'" >> "${log_file}" 2>&1

    local analyze_status=$?
    if [[ ${analyze_status} -ne 0 ]]; then
      echo "[WARN] analysis failed for ${tag} exit=${analyze_status}" | tee -a "${log_file}"
    else
      echo "[OK] analysis complete for ${tag}" | tee -a "${log_file}"
    fi
  else
    echo "[WARN] missing delta CSVs for ${tag}; skipping analysis" | tee -a "${log_file}"
  fi

  echo "[DONE] ${tag}" | tee -a "${log_file}"
  return 0
}

fail_count=0

# -----------------------------
# Pythia-70m (all layers)
# -----------------------------
for layer in 0 1 2 3 4 5; do
  run_case "EleutherAI/pythia-70m" "${layer}" "step1 step1000 step5000 step10000 step100000 steplatest" "48G" "02:30:00" || fail_count=$((fail_count+1))
done

# -----------------------------
# Pythia-1.4b (all 24 layers)
# -----------------------------
for layer in $(seq 0 23); do
  run_case "EleutherAI/pythia-1.4b" "${layer}" "step1 step1000 step5000 step10000 step100000 steplatest" "96G" "06:00:00" || fail_count=$((fail_count+1))
done

# -----------------------------
# OLMo-1B-hf (all 16 layers)
# Use available checkpoint names from repo docs.
# -----------------------------
for layer in $(seq 0 15); do
  run_case "allenai/OLMo-1B-hf" "${layer}" "step1000-tokens4B step343000-tokens1438B step425000-tokens1781B step509000-tokens2134B step593000-tokens2486B step738020-tokens3094B" "96G" "05:00:00" || fail_count=$((fail_count+1))
done

echo "Extensive ablation sweep finished with fail_count=${fail_count}. Logs: ${LOG_DIR}"
exit 0
