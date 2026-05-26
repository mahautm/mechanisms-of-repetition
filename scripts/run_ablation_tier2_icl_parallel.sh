#!/usr/bin/env bash
set -u

# Tier 2: Full Pythia-70m with both Natural AND ICL conditions.
# Submits all layers in PARALLEL using srun in background mode.
# Optional companion to extensive matrix—can run in parallel for faster completion.

ROOT_DIR="/home/mmahaut/projects/parrots"
OUT_DIR="${ROOT_DIR}/outputs_ablation_head_cycle"
LOG_DIR="${ROOT_DIR}/logs/ablation_tier2_icl"
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
  local tag="${safe_model}_L${layer}_icl_natural"
  local log_file="${LOG_DIR}/${tag}.log"

  cat > "${log_file}" << EOF
=================================================================
[START] model=${model_name} layer=${layer} conditions=natural+icl
checkpoints=${checkpoints}
=================================================================
EOF

  # Submit job in background
  srun "${COMMON_SRUN[@]}" --mem="${mem}" --time="${time_budget}" \
    bash -lc "source ~/.bashrc && conda activate parr && python '${ROOT_DIR}/ablation_head_cycle_icl_natural.py' \
      --model_name '${model_name}' \
      --checkpoints ${checkpoints} \
      --layer '${layer}' \
      --n_samples '${N_SAMPLES}' \
      --batch_size '${BATCH_SIZE}' \
      --max_prompt_tokens '${MAX_PROMPT_TOKENS}' \
      --max_new_tokens '${MAX_NEW_TOKENS}' \
      --seed '${SEED}' \
      --natural_dataset '${DATASET}' \
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
# SUBMIT ALL PYTHIA-70M JOBS (BOTH CONDITIONS) IN PARALLEL
# ===================================================================

echo "=========================================="
echo "TIER 2: Pythia-70m with ICL+Natural"
echo "Submitting all layers (each job runs BOTH conditions internally) in parallel..."
echo "=========================================="

# One job per layer; script computes natural+ICL in same run
for layer in 0 1 2 3 4 5; do
  echo "→ Layer ${layer}: queuing natural+ICL job"
  submit_job "EleutherAI/pythia-70m" "${layer}" "step1 step1000 step5000 step10000 step100000 steplatest" "48G" "02:30:00"
done

total_jobs=${#PIDS[@]}
echo ""
echo "=========================================="
echo "✓ Submitted ${total_jobs} jobs (6 layers, each with natural+ICL)"
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

for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  tag=${TAGS[$i]}
  log_file="${LOG_DIR}/${tag}.log"

  # Wait for this background job
  if wait "${pid}" 2>/dev/null; then
    echo "✓ [DONE] ${tag}"
    success_count=$((success_count + 1))
  else
    status=$?
    echo "✗ [FAIL] ${tag} (exit=${status})"
    fail_count=$((fail_count + 1))
  fi
done

echo ""
echo "=========================================="
echo "TIER 2 RESULTS"
echo "=========================================="
echo "Total jobs:   ${total_jobs}"
echo "Succeeded:    ${success_count}"
echo "Failed:       ${fail_count}"
echo "Logs:         ${LOG_DIR}"
echo "=========================================="
echo ""
echo "After completion, regenerate plots with:"
echo "  python aggregate_ablation_pythia70m.py"
echo "  python generate_ablation_conclusions.py"
echo "=========================================="

exit "${fail_count}"
