#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mmahaut/projects/parrots"
WORKER="$ROOT/run_ablation_all_models_1k_array_worker.sh"
N_SAMPLES="${N_SAMPLES:-1000}"
RAW_POOL_SIZE="${RAW_POOL_SIZE:-1000}"
RAW_PROMPT_TOKENS="${RAW_PROMPT_TOKENS:-32}"
PROMPT_GENERATION_TOKENS="${PROMPT_GENERATION_TOKENS:-1000}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1000}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs_ablation_head_cycle_1k_pile}"
STEP1_ONLY="${STEP1_ONLY:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"

mkdir -p "$ROOT/logs"

declare -A MODEL_NAME
declare -A LAYERS
declare -A BATCH_SIZE
declare -A USE_BNB
declare -A CHECKPOINTS
declare -A OUT_DIR
declare -A PLOTS_DIR
declare -A CHUNK_COUNT

MODEL_NAME[pythia70m]="EleutherAI/pythia-70m"
LAYERS[pythia70m]=6
BATCH_SIZE[pythia70m]=16
USE_BNB[pythia70m]=0
CHUNK_COUNT[pythia70m]=4
CHECKPOINTS[pythia70m]="step1|step1000|step36000|step71000|step143000"
OUT_DIR[pythia70m]="$OUT_ROOT/pythia70m"
PLOTS_DIR[pythia70m]="$ROOT/plots/ablation_pythia70m"

MODEL_NAME[pythia14b]="EleutherAI/pythia-1.4b"
LAYERS[pythia14b]=24
BATCH_SIZE[pythia14b]=8
USE_BNB[pythia14b]=1
CHUNK_COUNT[pythia14b]=4
CHECKPOINTS[pythia14b]="step1|step1000|step36000|step71000|step143000"
OUT_DIR[pythia14b]="$OUT_ROOT/pythia14b"
PLOTS_DIR[pythia14b]="$ROOT/plots/ablation_pythia14b"

MODEL_NAME[olmo1b]="allenai/OLMo-1B-hf"
LAYERS[olmo1b]=16
BATCH_SIZE[olmo1b]=8
USE_BNB[olmo1b]=1
CHUNK_COUNT[olmo1b]=4
CHECKPOINTS[olmo1b]="step1000-tokens4B|step5000-tokens20B|step117850-tokens494B|step369000-tokens1547B|step738020-tokens3094B"
OUT_DIR[olmo1b]="$OUT_ROOT/olmo1b"
PLOTS_DIR[olmo1b]="$ROOT/plots/ablation_olmo1b"

MODEL_NAME[apertus8b]="swiss-ai/Apertus-8B-2509"
LAYERS[apertus8b]=32
BATCH_SIZE[apertus8b]=24
USE_BNB[apertus8b]=1
CHUNK_COUNT[apertus8b]=4
CHECKPOINTS[apertus8b]="step50000-tokens210B|step650000-tokens2730B|step1432000-tokens6014B|step2627139-tokens15T"
OUT_DIR[apertus8b]="$OUT_ROOT/apertus8b"
PLOTS_DIR[apertus8b]="$ROOT/plots/ablation_apertus8b"

ORDER=(pythia70m pythia14b olmo1b apertus8b)
POST_JOBS=()

echo "Launching all-model ablation reruns with N_SAMPLES=$N_SAMPLES, RAW_POOL_SIZE=$RAW_POOL_SIZE, MAX_NEW_TOKENS=$MAX_NEW_TOKENS, STEP1_ONLY=$STEP1_ONLY"
echo "Array concurrency cap: $MAX_CONCURRENT"

for tag in "${ORDER[@]}"; do
  model="${MODEL_NAME[$tag]}"
  layers="${LAYERS[$tag]}"
  batch="${BATCH_SIZE[$tag]}"
  use_bnb="${USE_BNB[$tag]}"
  chunk_count="${CHUNK_COUNT[$tag]}"
  checkpoints="${CHECKPOINTS[$tag]}"
  if [[ "$STEP1_ONLY" == "1" ]]; then
    checkpoints="${checkpoints%%|*}"
  fi
  checkpoints_csv="${checkpoints//|/,}"
  out_dir="${OUT_DIR[$tag]}"
  plots_dir="${PLOTS_DIR[$tag]}"
  if [[ "$STEP1_ONLY" == "1" ]]; then
    out_dir="${out_dir}_step1"
    plots_dir="${plots_dir}_step1"
  fi
  IFS='|' read -r -a ckpts_arr <<< "$checkpoints"
  ckpt_count="${#ckpts_arr[@]}"
  total_tasks=$((layers * ckpt_count * chunk_count))
  safe_model="${model//\//_}"
  merged_raw="$out_dir/head_cycle_ablation_${safe_model}_ALL_icl_natural.csv"

  rm -rf "$out_dir"
  mkdir -p "$out_dir" "$plots_dir"

  array_job=$(sbatch --parsable \
    --job-name="abl1k_${tag}" \
    --array="0-$((total_tasks-1))%$MAX_CONCURRENT" \
    --export=ALL,MODEL_NAME="$model",CHECKPOINTS_CSV="$checkpoints",LAYER_COUNT="$layers",CHUNK_COUNT="$chunk_count",N_SAMPLES="$N_SAMPLES",RAW_POOL_SIZE="$RAW_POOL_SIZE",RAW_PROMPT_TOKENS="$RAW_PROMPT_TOKENS",PROMPT_GENERATION_TOKENS="$PROMPT_GENERATION_TOKENS",BATCH_SIZE="$batch",MAX_PROMPT_TOKENS="$MAX_PROMPT_TOKENS",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",OUT_DIR="$out_dir",USE_BNB="$use_bnb" \
    "$WORKER")

  post_cmd="srun --partition=alien --qos=alien --exclude=node044 /bin/bash -lc 'source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && python $ROOT/merge_ablation_layers_generic.py --input_dir $out_dir --model_name $model --output_csv $merged_raw --expected_layers $layers --expected_checkpoints $checkpoints_csv && python $ROOT/build_combined_ablation_generic.py --input_csv $merged_raw --output_dir $plots_dir'"

  post_job=$(sbatch --parsable --job-name="abl1k_post_${tag}" --dependency="afterok:${array_job}" --wrap="$post_cmd")
  POST_JOBS+=("$post_job")

  echo "[$tag] array job: $array_job"
  echo "[$tag] post job:  $post_job"
done

replot_cmd="srun --partition=alien --qos=alien --exclude=node044 /bin/bash -lc 'source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && cd $ROOT && python plot_user_friendly_scatter.py --discover --metric delta_repetition_rate --jitter 0.002 --all-checkpoints && python plot_user_friendly_scatter.py --discover --metric delta_avg_cycle_count --jitter 0.002 --all-checkpoints && python plot_checkpoint_comparison_panel.py'"

dep_joined="$(IFS=:; echo "${POST_JOBS[*]}")"
replot_job=$(sbatch --parsable --job-name="abl1k_replot" --dependency="afterok:${dep_joined}" --wrap="$replot_cmd")

echo "[replot] dependency: afterok:${dep_joined}"
echo "[replot] job: $replot_job"
