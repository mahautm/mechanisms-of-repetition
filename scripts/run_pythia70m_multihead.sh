#!/bin/bash
#SBATCH --job-name=pythia70m_multihead
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pythia70m_multihead_%A_%a.out
#SBATCH --error=logs/pythia70m_multihead_%A_%a.err
#SBATCH --array=0-5

# Pythia-70m has 6 layers, so we analyze all of them
# Checkpoints: step1, step1000, step5000, step10000, step100000, steplatest

# Define checkpoints
CHECKPOINTS=("step1" "step1000" "step5000" "step10000" "step100000" "steplatest")
CHECKPOINT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

# Model Configuration
MODEL_NAME="EleutherAI/pythia-70m"
CLEAN_MODEL_NAME=$(echo ${MODEL_NAME} | tr '/' '_')
MAX_LAYERS=6  # Pythia-70m has 6 layers

# Analysis Parameters
N_CYCLES=4
MAX_LENGTH=32
BATCH_SIZE=16  # Smaller model, can use larger batch
N_SAMPLES=100
MAX_NEW_TOKENS=1000

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/outputs_multihead_full"

echo "================================================"
echo "Pythia-70m Multi-head Analysis"
echo "Checkpoint: ${CHECKPOINT}"
echo "================================================"

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0

cd ~/projects/parrots/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run analysis for ALL layers at this checkpoint
for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
    echo "Analyzing layer ${layer_idx}/${MAX_LAYERS} for checkpoint ${CHECKPOINT}..."
    
    SPECIFIC_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}/${CHECKPOINT}/layer_${layer_idx}"
    mkdir -p ${SPECIFIC_OUTPUT}
    
    REVISION_FLAG=""
    if [ "${CHECKPOINT}" != "steplatest" ]; then
        REVISION_FLAG="--revision=${CHECKPOINT}"
    fi
    
    poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/ckpt_pipeline_main.py \
        --model-name="${MODEL_NAME}" \
        --single-lens=${layer_idx} \
        --max-layer-idx=${MAX_LAYERS} \
        --lens-path="/home/mmahaut/projects/parrots/lenses_multihead/${CLEAN_MODEL_NAME}/" \
        --n-cycles=${N_CYCLES} \
        --max-length=${MAX_LENGTH} \
        --max-new-tokens=${MAX_NEW_TOKENS} \
        --batch-size=${BATCH_SIZE} \
        --n-samples=${N_SAMPLES} \
        ${REVISION_FLAG}
    
    # Move outputs to layer-specific directory
    mv *.png "${SPECIFIC_OUTPUT}/" 2>/dev/null || true
    mv *.npy "${SPECIFIC_OUTPUT}/" 2>/dev/null || true
    mv *.txt "${SPECIFIC_OUTPUT}/" 2>/dev/null || true
    
    echo "Completed layer ${layer_idx} for checkpoint ${CHECKPOINT}"
done

echo "================================================"
echo "Completed all layers for checkpoint ${CHECKPOINT}"
echo "================================================"
