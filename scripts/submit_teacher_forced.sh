#!/bin/bash
# Submit the teacher-forced ablation for one model: baseline (per checkpoint), then single-head
# and group ablations once the baselines exist.
#   scripts/submit_teacher_forced.sh                       # Pythia-1.4B, all six checkpoints
#   CHECKPOINTS="step143000" scripts/submit_teacher_forced.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-1.4b}"
TAG="${TAG:-pythia14b}"
LAYER_COUNT="${LAYER_COUNT:-24}"
LAYERS_PER_TASK="${LAYERS_PER_TASK:-12}"  # few large tasks: the cluster-wide job cap is easily hit
ABLATION="${ABLATION:-zero}"
CHECKPOINTS="${CHECKPOINTS:-step0|step1|step1000|step36000|step71000|step143000}"
OUT_DIR="${OUT_DIR:-outputs_teacher_forced/$TAG}"
IFS='|' read -r -a C <<< "$CHECKPOINTS"
N=${#C[@]}
BLOCKS=$(( (LAYER_COUNT + LAYERS_PER_TASK - 1) / LAYERS_PER_TASK ))
COMMON="MODEL_NAME=$MODEL_NAME,CHECKPOINTS_CSV=$CHECKPOINTS,OUT_DIR=$OUT_DIR,ABLATION=$ABLATION"

set -o pipefail
base=$(sbatch --parsable --job-name="tf_base_$TAG" --array=0-$((N - 1)) \
  --export=ALL,STAGE=baseline,$COMMON slurm/teacher_forced_worker.sh)
single=$(sbatch --parsable --job-name="tf_single_$TAG" --time=12:00:00 --dependency=afterok:$base --array=0-$((N * BLOCKS - 1)) \
  --export=ALL,STAGE=single,LAYER_COUNT=$LAYER_COUNT,LAYERS_PER_TASK=$LAYERS_PER_TASK,$COMMON slurm/teacher_forced_worker.sh)
group=$(sbatch --parsable --job-name="tf_group_$TAG" --dependency=afterok:$base --array=0-$((N - 1)) \
  --export=ALL,STAGE=group,$COMMON slurm/teacher_forced_worker.sh)
echo "baseline=$base single=$single group=$group -> $OUT_DIR"
