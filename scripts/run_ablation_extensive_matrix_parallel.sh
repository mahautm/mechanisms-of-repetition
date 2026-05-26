#!/usr/bin/env bash
set -u

# Extensive head-by-head ablation sweep across all models/layers/checkpoints.
# Submits all jobs in PARALLEL using srun in background mode.
# Saturates alien partition with many concurrent jobs.

ROOT_DIR="/home/mmahaut/projects/parrots"
OUT_DIR="${ROOT_DIR}/outputs_ablation_head_cycle"
LOG_DIR="${ROOT_DIR}/logs/ablation_extensive"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

CONDITIONS_MODE="${CONDITIONS_MODE:-natural}"
if [[ "${CONDITIONS_MODE}" != "natural" && "${CONDITIONS_MODE}" != "icl_natural" ]]; then
  echo "[ERROR] CONDITIONS_MODE must be 'natural' or 'icl_natural' (got '${CONDITIONS_MODE}')"
  exit 2
fi

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

declare -a PIDS
declare -a TAGS

submit_job() {
  local model_name="$1"
  local layer="$2"
  local checkpoints="$3"
  local mem="$4"
  local time_budget="$5"

  local safe_model
  safe_model="$(echo "${model_name}" | tr '/' '_')"
  local tag="${safe_model}_L${layer}"
  local log_file="${LOG_DIR}/${tag}.log"

  cat > "${log_file}" << EOF
=================================================================
[START] model=${model_name} layer=${layer} checkpoints=${checkpoints}
=================================================================
EOF

  local ablation_script="${ROOT_DIR}/ablation_head_cycle_evolution.py"
  local condition_args="--dataset '${DATASET}' --cycle_sizes 2,3,4,5,6"
  if [[ "${CONDITIONS_MODE}" == "icl_natural" ]]; then
    ablation_script="${ROOT_DIR}/ablation_head_cycle_icl_natural.py"
    condition_args="--natural_dataset '${DATASET}'"
  fi

  # Submit ablation job in background
  srun "${COMMON_SRUN[@]}" --mem="${mem}" --time="${time_budget}" \
    bash -lc "source ~/.bashrc && conda activate parr && python '${ablation_script}' \
      --model_name '${model_name}' \
      --checkpoints ${checkpoints} \
      --layer '${layer}' \
      --n_samples '${N_SAMPLES}' \
      --batch_size '${BATCH_SIZE}' \
      --max_prompt_tokens '${MAX_PROMPT_TOKENS}' \
      --max_new_tokens '${MAX_NEW_TOKENS}' \
      --seed '${SEED}' \
      ${condition_args} \
      --do_sample \
      --temperature 0.8 \
      --top_p 0.9 \
      --output_dir '${OUT_DIR}'" >> "${log_file}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  TAGS+=("${tag}")
  echo "[QUEUED] ${tag} with PID ${pid}" | tee -a "${log_file}"
}

# ===================================================================
# SUBMIT ALL JOBS IMMEDIATELY (NO BLOCKING)
# ===================================================================

echo "=========================================="
echo "Submitting ALL jobs in parallel..."
echo "Condition mode: ${CONDITIONS_MODE}"
echo "=========================================="

# Pythia-70m (all 6 layers)
echo "→ Pythia-70m: 6 layers"
for layer in 0 1 2 3 4 5; do
  submit_job "EleutherAI/pythia-70m" "${layer}" "step1 step1000 step5000 step10000 step100000 steplatest" "48G" "02:30:00"
done

# Pythia-1.4b (all 24 layers)
echo "→ Pythia-1.4b: 24 layers"
for layer in $(seq 0 23); do
  submit_job "EleutherAI/pythia-1.4b" "${layer}" "step1 step1000 step5000 step10000 step100000 steplatest" "96G" "06:00:00"
done

# OLMo-1B (all 16 layers)
echo "→ OLMo-1B: 16 layers"
for layer in $(seq 0 15); do
  submit_job "allenai/OLMo-1B-hf" "${layer}" "step1000-tokens4B step343000-tokens1438B step425000-tokens1781B step509000-tokens2134B step593000-tokens2486B step738020-tokens3094B" "96G" "05:00:00"
done

total_jobs=${#PIDS[@]}
echo ""
echo "=========================================="
echo "✓ Submitted ${total_jobs} jobs to alien partition"
echo "✓ All running in parallel—cluster will auto-schedule"
echo "=========================================="
echo ""

# ===================================================================
# MONITOR ALL JOBS
# ===================================================================

echo "Waiting for all jobs to complete..."
echo "Log directory: ${LOG_DIR}"
echo ""

fail_count=0
success_count=0
timeout_count=0

for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  tag=${TAGS[$i]}
  log_file="${LOG_DIR}/${tag}.log"

  # Wait for this background job
  if wait "${pid}" 2>/dev/null; then
    echo "✓ [DONE] ${tag}"
    success_count=$((success_count + 1))

    # Run analysis on results
    safe_model="$(echo "${tag}" | sed 's/_L.*$//')"
    layer="${tag##*_L}"
    overall_csv="${OUT_DIR}/head_cycle_ablation_${safe_model}_L${layer}_overall_delta.csv"
    cycle_csv="${OUT_DIR}/head_cycle_ablation_${safe_model}_L${layer}_cycle_delta.csv"

    if [[ -f "${overall_csv}" && -f "${cycle_csv}" ]]; then
      # Run analysis (quick, non-blocking)
      srun --partition=alien --qos=alien --exclude=node044 --time=00:15:00 --mem=8G \
        bash -lc "source ~/.bashrc && conda activate parr && python '${ROOT_DIR}/analyze_ablation_head_cycle.py' \
          --overall_delta_csv '${overall_csv}' \
          --cycle_delta_csv '${cycle_csv}' \
          --output_dir '${OUT_DIR}'" >> "${log_file}" 2>&1 &
    fi
  else
    status=$?
    echo "✗ [FAIL] ${tag} (exit=${status})" | tee -a "${log_file}"
    fail_count=$((fail_count + 1))
  fi
done

echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
echo "Total jobs:   ${total_jobs}"
echo "Succeeded:    ${success_count}"
echo "Failed:       ${fail_count}"
echo "Logs:         ${LOG_DIR}"
echo "=========================================="

exit "${fail_count}"
