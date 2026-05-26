#!/bin/bash

# Train MLP lenses for all layers using the same pattern as train_multihead_lenses.sh
# This script follows the working SLURM pattern

# SLURM Parameters
MEM="120G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien" 
EXCLUDE=""

# Model Configuration
MODEL_NAME="EleutherAI/pythia-1.4b"
MAX_LAYERS=24

# Training Parameters
EPOCHS=10
LR="1e-3"
TRAIN_SAMPLES=10000
BATCH_SIZE=8
MAX_LENGTH=512

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/lenses_mlp"
mkdir -p ${OUTPUT_DIR}

echo "Starting MLP lens training for ${MODEL_NAME}"
echo "Training MLP lenses across ${MAX_LAYERS} layers"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Train lenses for each layer
for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
    
    JOB_NAME="mlp_lens_layer_${layer_idx}"
    LOG_DIR="${OUTPUT_DIR}/logs"
    mkdir -p ${LOG_DIR}
    ERROR_LOG="${LOG_DIR}/layer_${layer_idx}.err"
    OUTPUT_LOG="${LOG_DIR}/layer_${layer_idx}.out"
    
    echo "Submitting job for layer ${layer_idx}..."
    
    # Create the training script
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

echo "Training MLP lens for layer ${layer_idx}"
echo "Model: ${MODEL_NAME}"
echo "Epochs: ${EPOCHS}, LR: ${LR}, Samples: ${TRAIN_SAMPLES}"

# Run training
poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/aa_fortu_train_mlp_lens.py \
    --model-name="${MODEL_NAME}" \
    --layer-idx=${layer_idx} \
    --epochs=${EPOCHS} \
    --lr=${LR} \
    --train-samples=${TRAIN_SAMPLES} \
    --output-dir="${OUTPUT_DIR}" \
    --batch-size=${BATCH_SIZE} \
    --max-length=${MAX_LENGTH}

echo "Completed training for layer ${layer_idx}"
echo "Output saved to: ${OUTPUT_DIR}/EleutherAI_pythia-1.4b/layer_${layer_idx}_mlp_lens.pth"
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
    
    echo "Submitted job ${JOB_NAME} for layer ${layer_idx}"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "All training jobs submitted!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: ${OUTPUT_DIR}/logs/"
echo ""
echo "Expected output files:"
for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
    echo "  ${OUTPUT_DIR}/EleutherAI_pythia-1.4b/layer_${layer_idx}_mlp_lens.pth"
done