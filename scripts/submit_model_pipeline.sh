#!/bin/bash
# Full pipeline for one model: teacher-forced baseline -> {single-head, group, onset} ablations.
#   MODEL_NAME=EleutherAI/pythia-70m TAG=pythia70m LAYER_COUNT=6 N_SETS=26 \
#     CHECKPOINTS="step0|step1000|step36000|step71000|step143000" scripts/submit_model_pipeline.sh
# N_SETS = 26 for models with < 13 layers (no L12H0 set), 27 otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
: "${MODEL_NAME:?}" "${TAG:?}" "${LAYER_COUNT:?}" "${CHECKPOINTS:?}"
N_SETS="${N_SETS:-27}"
ABLATION="${ABLATION:-zero}"
IFS='|' read -r -a C <<< "$CHECKPOINTS"
N=${#C[@]}
PER=$(( LAYER_COUNT < 12 ? LAYER_COUNT : 12 ))
BLOCKS=$(( (LAYER_COUNT + PER - 1) / PER ))
TF="outputs_teacher_forced/$TAG"
COMMON="MODEL_NAME=$MODEL_NAME,CHECKPOINTS_CSV=$CHECKPOINTS,OUT_DIR=$TF,ABLATION=$ABLATION"
sub() { local id; for _ in 1 2 3 4 5; do id=$(sbatch --parsable "$@" 2>/dev/null) && { echo "$id"; return; }; sleep 30; done; echo "submit failed: $*" >&2; exit 1; }
base=$(sub --job-name="tf_base_$TAG" --array=0-$((N - 1)) --export=ALL,STAGE=baseline,$COMMON slurm/teacher_forced_worker.sh)
single=$(sub --job-name="tf_single_$TAG" --time=12:00:00 --dependency=afterok:$base --array=0-$((N * BLOCKS - 1)) \
  --export=ALL,STAGE=single,LAYER_COUNT=$LAYER_COUNT,LAYERS_PER_TASK=$PER,$COMMON slurm/teacher_forced_worker.sh)
group=$(sub --job-name="tf_group_$TAG" --dependency=afterok:$base --array=0-$((N - 1)) --export=ALL,STAGE=group,$COMMON slurm/teacher_forced_worker.sh)
onset=$(sub --job-name="onset_$TAG" --dependency=afterok:$base --array=0-$((N * 3 - 1)) \
  --export="ALL,CHECKPOINTS_CSV=$CHECKPOINTS,N_SETS=$N_SETS,SETS_PER_TASK=$(( (N_SETS + 2) / 3 )),EXTRA_ARGS=--model_name $MODEL_NAME --scores_dir $TF --output_dir outputs_onset/$TAG --ablation $ABLATION" \
  slurm/onset_worker.sh)
echo "$TAG: baseline=$base single=$single group=$group onset=$onset"
