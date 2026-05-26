#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mmahaut/projects/parrots"
ARRAY_SCRIPT="$ROOT/run_apertus_ablation_all_layers_array.sh"
RAW_OUT_DIR="$ROOT/outputs_ablation_head_cycle_apertus_all_layers"
MERGED_RAW="$RAW_OUT_DIR/head_cycle_ablation_swiss-ai_Apertus-8B-2509_ALL_icl_natural.csv"
PLOTS_DIR="$ROOT/plots/ablation_apertus8b"

mkdir -p "$ROOT/logs" "$RAW_OUT_DIR" "$PLOTS_DIR"

ARRAY_JOB=$(sbatch --parsable "$ARRAY_SCRIPT")
echo "Submitted array job: $ARRAY_JOB"

POST_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} --wrap="srun --partition=alien --qos=alien --exclude=node044 /bin/bash -c 'source ~/.bashrc >/dev/null 2>&1 || true; conda activate parr && python $ROOT/merge_apertus_ablation_layers.py --input_dir $RAW_OUT_DIR --output_csv $MERGED_RAW && python $ROOT/build_apertus_combined_ablation.py --input_csv $MERGED_RAW --output_dir $PLOTS_DIR && python $ROOT/plot_user_friendly_scatter.py --discover'")

echo "Submitted post-processing job: $POST_JOB"
echo "Track array: squeue -j ${ARRAY_JOB}"
echo "Track post:  squeue -j ${POST_JOB}"
