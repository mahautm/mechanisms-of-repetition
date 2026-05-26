#!/usr/bin/env bash
# Launcher to resubmit missing alluvial tasks with fewer GPUs per job.
# Scans outputs and only submits jobs for missing batch ranges.

set -euo pipefail
source ~/.bashrc
conda activate parr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT_DIR="/home/mmahaut/projects/parrots"
OUT_ROOT="${ROOT_DIR}/outputs_multihead_full_new"
MODEL_NAME="EleutherAI/pythia-1.4b"
LAYER=23
N_SAMPLES=512
BATCHES_PER_CHECKPOINT=16
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
TOTAL_TASKS=$(( ${#CHECKPOINTS[@]} * BATCHES_PER_CHECKPOINT ))

# Configure resources for few-GPU regime
GPUS_PER_JOB=${1:-1}   # default 1 GPU if not provided
MEM_PER_JOB=${2:-64G}
TIME_PER_JOB=${3:-04:00:00}
CONCURRENCY=${4:-32}

# device_map decision
if [ "$GPUS_PER_JOB" -le 1 ]; then
  DEVICE_MAP_ARG="none"
else
  DEVICE_MAP_ARG="auto"
fi

echo "Resubmitting missing tasks for model=${MODEL_NAME}, layer=${LAYER}, gpuset=${GPUS_PER_JOB}, device_map=${DEVICE_MAP_ARG}"

tasks_to_submit=()

for (( task_id=0; task_id<TOTAL_TASKS; task_id++ )); do
  chkpt_index=$(( task_id / BATCHES_PER_CHECKPOINT ))
  batch_index=$(( task_id % BATCHES_PER_CHECKPOINT ))
  chkpt=${CHECKPOINTS[$chkpt_index]}
  sample_offset=$(( batch_index * N_SAMPLES ))

  out_dir="${OUT_ROOT}/${MODEL_NAME}/${chkpt}/layer_${LAYER}"
  length_tag="full"
  offset_tag="off${sample_offset}"
  out_file="${out_dir}/full_analysis_cyc4_${length_tag}_${offset_tag}.out"

  if [ -f "$out_file" ]; then
    # already done
    continue
  else
    tasks_to_submit+=("$task_id")
  fi
done

if [ ${#tasks_to_submit[@]} -eq 0 ]; then
  echo "All tasks complete — nothing to submit."
  exit 0
fi

echo "Found ${#tasks_to_submit[@]} missing tasks. Submitting jobs with ${GPUS_PER_JOB} GPU(s) each..."

# Submit each missing task as an individual sbatch --wrap job to simplify control
for tid in "${tasks_to_submit[@]}"; do
  chkpt_index=$(( tid / BATCHES_PER_CHECKPOINT ))
  batch_index=$(( tid % BATCHES_PER_CHECKPOINT ))
  chkpt=${CHECKPOINTS[$chkpt_index]}
  sample_offset=$(( batch_index * N_SAMPLES ))

  job_name="parrots-alluvial-${chkpt}-b${batch_index}"
  log_out="${ROOT_DIR}/logs/${job_name}.out"
  cmd="python ${ROOT_DIR}/generate_alluvial_data.py --model-name \"${MODEL_NAME}\" --revision \"${chkpt}\" --n-samples ${N_SAMPLES} --sample-offset ${sample_offset} --batch-size 1 --max-length 0 --max-new-tokens 1000 --n-cycles 2 --layer ${LAYER} --device-map ${DEVICE_MAP_ARG} --output-dir \"${OUT_ROOT}\""

  echo "Submitting task ${tid}: checkpoint=${chkpt} batch=${batch_index} offset=${sample_offset}"
  # Name the job and submit on the alien partition/QoS, excluding node044
  sbatch --parsable --job-name=${job_name} --partition=alien --qos=alien --exclude=node044 --gres=gpu:${GPUS_PER_JOB} --mem=${MEM_PER_JOB} --time=${TIME_PER_JOB} --cpus-per-task=4 --output=${log_out} --wrap "${cmd}"
  sleep 0.2
done

echo "Submitted ${#tasks_to_submit[@]} jobs."