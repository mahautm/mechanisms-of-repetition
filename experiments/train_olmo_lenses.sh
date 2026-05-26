#!/bin/bash

# Train multi-head lenses for OLMo models
# Trains separate lenses for each attention head in layer 12

# SLURM Parameters
MEM="120G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien" 
EXCLUDE="node044"

# Model Configuration - select which model to train
MODEL_NAME="${1:-allenai/OLMo-1B-hf}"
REVISION="${2:-}"  # Optional: specific checkpoint

# Determine model parameters based on model name
if [[ "$MODEL_NAME" == *"OLMo"* ]]; then
    MAX_LAYERS=16
    NUM_HEADS=16
    HEAD_DIM=128
else
    echo "Error: Unknown model architecture for $MODEL_NAME"
    exit 1
fi

# Training layer - just train layer 12 for now
TRAIN_LAYER=12

# Training Parameters
EPOCHS=10
LR="1e-3"
TRAIN_SAMPLES=10000
BATCH_SIZE=8
MAX_LENGTH=512

# Output directory
if [ -n "$REVISION" ]; then
    OUTPUT_DIR="/home/mmahaut/projects/parrots/lenses_multihead/${MODEL_NAME/\//_}/${REVISION}"
else
    OUTPUT_DIR="/home/mmahaut/projects/parrots/lenses_multihead/${MODEL_NAME/\//_}"
fi
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "OLMo Multi-Head Lens Training"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
if [ -n "$REVISION" ]; then
    echo "Checkpoint: ${REVISION}"
fi
echo "Training layer: ${TRAIN_LAYER}"
echo "Heads per layer: ${NUM_HEADS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Create log directory
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}

JOB_NAME="olmo_lens_l${TRAIN_LAYER}"
ERROR_LOG="${LOG_DIR}/layer_${TRAIN_LAYER}.err"
OUTPUT_LOG="${LOG_DIR}/layer_${TRAIN_LAYER}.out"

echo "Submitting job for layer ${TRAIN_LAYER}..."

# Create the training script
if [ -n "$REVISION" ]; then
    REVISION_ARG="--revision=\"${REVISION}\""
else
    REVISION_ARG=""
fi

script=$(cat <<EOF
echo "SLURM Job Parameters:" > sbatch_params.txt
echo "SBATCH_JOB_ID=\$SLURM_JOB_ID" >> sbatch_params.txt
echo "SBATCH_JOB_NAME=\$SLURM_JOB_NAME" >> sbatch_params.txt
echo "SLURMD_NODENAME=\$SLURMD_NODENAME" >> sbatch_params.txt

source ~/.bashrc
echo "Running on node: \$SLURMD_NODENAME"
echo "GPU devices: \$CUDA_VISIBLE_DEVICES"

# Activate environment
conda activate parr
module load CUDA/12.2.0

# Check Python and CUDA
which python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Change to project directory
cd ~/projects/parrots/

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Training multi-head lens for layer ${TRAIN_LAYER}"
echo "Model: ${MODEL_NAME}"
if [ -n "${REVISION}" ]; then
    echo "Checkpoint: ${REVISION}"
fi
echo "Epochs: ${EPOCHS}, LR: ${LR}, Samples: ${TRAIN_SAMPLES}"
echo "Heads: ${NUM_HEADS}, Head dim: ${HEAD_DIM}"

# Run training
python /home/mmahaut/projects/parrots/parrots/aa_fortu/aa_fortu_train_multihead_lens.py \
    --model-name="${MODEL_NAME}" \
    ${REVISION_ARG} \
    --layer-idx=${TRAIN_LAYER} \
    --epochs=${EPOCHS} \
    --lr=${LR} \
    --train-samples=${TRAIN_SAMPLES} \
    --output-dir="${OUTPUT_DIR}" \
    --batch-size=${BATCH_SIZE} \
    --max-length=${MAX_LENGTH} \
    --num-heads=${NUM_HEADS} \
    --head-dim=${HEAD_DIM}

echo "Completed training for layer ${TRAIN_LAYER}"
echo "Output saved to: ${OUTPUT_DIR}/layer_${TRAIN_LAYER}_multihead_lens.pth"
EOF
)

# Submit the job
sbatch \
    --job-name=${JOB_NAME} \
    --output=${OUTPUT_LOG} \
    --error=${ERROR_LOG} \
    --mem=${MEM} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --exclude=${EXCLUDE} \
    --gres=${GRES} \
    --nice=10 \
    --wrap="$script"

echo ""
echo "=========================================="
echo "Training job submitted!"
echo "=========================================="
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: ${OUTPUT_DIR}/logs/"
echo ""
echo "Expected output file:"
echo "  ${OUTPUT_DIR}/layer_${TRAIN_LAYER}_multihead_lens.pth"
echo "=========================================="
