#!/usr/bin/env bash
#SBATCH --job-name=parrots-alluvial-ckpt
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --array=0-127%32  # 8 checkpoints × 16 batches = 128 tasks (concurrency 32)
#SBATCH --output=/home/mmahaut/projects/parrots/logs/alluvial_ckpt_%A_%a.out

source ~/.bashrc
conda activate parr
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT_DIR="/home/mmahaut/projects/parrots"

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

# Determine checkpoint and batch index from flattened array index
TASK_ID=${SLURM_ARRAY_TASK_ID}
BATCHES_PER_CHECKPOINT=16
N_SAMPLES=512

CHKPT_INDEX=$(( TASK_ID / BATCHES_PER_CHECKPOINT ))
BATCH_INDEX=$(( TASK_ID % BATCHES_PER_CHECKPOINT ))

CHKPT="${CHECKPOINTS[$CHKPT_INDEX]}"
SAMPLE_OFFSET=$(( BATCH_INDEX * N_SAMPLES ))

echo "[START] Task $TASK_ID -> pythia-1.4b @ Checkpoint=$CHKPT (batch $BATCH_INDEX offset $SAMPLE_OFFSET)"

python "${ROOT_DIR}/generate_alluvial_data.py" \
    --model-name "EleutherAI/pythia-1.4b" \
    --revision "${CHKPT}" \
    --n-samples ${N_SAMPLES} \
    --sample-offset ${SAMPLE_OFFSET} \
    --batch-size 1 \
    --max-length 0 \
    --max-new-tokens 1000 \
    --n-cycles 2 \
    --layer 23 \
    --device-map auto \
    --output-dir "${ROOT_DIR}/outputs_multihead_full_new"

echo "[DONE] Task $TASK_ID"
