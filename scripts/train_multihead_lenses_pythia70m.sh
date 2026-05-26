#!/bin/bash

# Train multi-head lenses for Pythia-70m (6 layers, 8 heads each)
# This trains linear maps from each attention head's output to vocabulary

# SLURM Parameters
MEM="64G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien" 
EXCLUDE=""

# Model Configuration - Pythia-70m specifics
MODEL_NAME="EleutherAI/pythia-70m"
MAX_LAYERS=6
NUM_HEADS=8
HEAD_DIM=64  # 512 hidden / 8 heads = 64

# Training Parameters
EPOCHS=10
LR="1e-3"
TRAIN_SAMPLES=10000
BATCH_SIZE=16  # Can use larger batch for smaller model
MAX_LENGTH=512

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/lenses_multihead"
CLEAN_MODEL_NAME=$(echo ${MODEL_NAME} | tr '/' '_')
mkdir -p ${OUTPUT_DIR}/${CLEAN_MODEL_NAME}
mkdir -p ${OUTPUT_DIR}/logs

echo "Starting multi-head lens training for ${MODEL_NAME}"
echo "Training ${NUM_HEADS} heads per layer across ${MAX_LAYERS} layers"
echo "Output directory: ${OUTPUT_DIR}/${CLEAN_MODEL_NAME}"

# Train lenses for each layer
for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
    
    JOB_NAME="mh_lens_70m_L${layer_idx}"
    LOG_DIR="${OUTPUT_DIR}/logs"
    ERROR_LOG="${LOG_DIR}/pythia70m_layer_${layer_idx}.err"
    OUTPUT_LOG="${LOG_DIR}/pythia70m_layer_${layer_idx}.out"
    
    echo "Submitting job for layer ${layer_idx}..."
    
    # Create the training script
    script=$(cat <<EOF
source ~/.bashrc
echo "Running on node: \$SLURMD_NODENAME"

# Activate environment
conda activate parr
module load CUDA/12.2.0

cd ~/projects/parrots/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Training multi-head lens for ${MODEL_NAME} layer ${layer_idx}"
echo "Heads: ${NUM_HEADS}, Head dim: ${HEAD_DIM}"

# Run training
poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/aa_fortu_train_multihead_lens.py \
    --model-name="${MODEL_NAME}" \
    --layer-idx=${layer_idx} \
    --epochs=${EPOCHS} \
    --lr=${LR} \
    --train-samples=${TRAIN_SAMPLES} \
    --output-dir="${OUTPUT_DIR}" \
    --batch-size=${BATCH_SIZE} \
    --max-length=${MAX_LENGTH} \
    --num-heads=${NUM_HEADS} \
    --head-dim=${HEAD_DIM}

echo "Completed training for layer ${layer_idx}"
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
        --gres=${GRES} \
        --time=4:00:00 \
        --wrap="$script"
    
    echo "Submitted ${JOB_NAME}"
    sleep 1
done

echo ""
echo "All ${MAX_LAYERS} training jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Logs: ${OUTPUT_DIR}/logs/pythia70m_layer_*.out"
