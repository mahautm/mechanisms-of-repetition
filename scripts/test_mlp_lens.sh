#!/bin/bash

# Test MLP lens training for a single layer using the working SLURM pattern

# SLURM Parameters
MEM="32G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien" 
EXCLUDE="node044"

# Model Configuration
MODEL_NAME="EleutherAI/pythia-1.4b"
TEST_LAYER=8

# Training Parameters (smaller for testing)
EPOCHS=2
LR="1e-3"
TRAIN_SAMPLES=1000
BATCH_SIZE=4
MAX_LENGTH=512

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/lenses_mlp_test"
mkdir -p ${OUTPUT_DIR}

echo "Testing MLP lens training for ${MODEL_NAME} layer ${TEST_LAYER}"
echo "Output directory: ${OUTPUT_DIR}"

JOB_NAME="test_mlp_lens_layer_${TEST_LAYER}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}
ERROR_LOG="${LOG_DIR}/test_layer_${TEST_LAYER}.err"
OUTPUT_LOG="${LOG_DIR}/test_layer_${TEST_LAYER}.out"

echo "Submitting test job for layer ${TEST_LAYER}..."

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

echo "Testing MLP lens training for layer ${TEST_LAYER}"
echo "Model: ${MODEL_NAME}"
echo "Epochs: ${EPOCHS}, LR: ${LR}, Samples: ${TRAIN_SAMPLES}"

# Run training
poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/aa_fortu_train_mlp_lens.py \
    --model-name="${MODEL_NAME}" \
    --layer-idx=${TEST_LAYER} \
    --epochs=${EPOCHS} \
    --lr=${LR} \
    --train-samples=${TRAIN_SAMPLES} \
    --output-dir="${OUTPUT_DIR}" \
    --batch-size=${BATCH_SIZE} \
    --max-length=${MAX_LENGTH}

EXIT_CODE=\$?

echo "Training completed with exit code: \$EXIT_CODE"

if [ \$EXIT_CODE -eq 0 ]; then
    LENS_FILE="${OUTPUT_DIR}/EleutherAI_pythia-1.4b/layer_${TEST_LAYER}_mlp_lens.pth"
    if [ -f "\$LENS_FILE" ]; then
        echo "✅ MLP lens saved successfully: \$LENS_FILE"
        echo "   File size: \$(du -h "\$LENS_FILE" | cut -f1)"
        
        # Test loading the lens
        python -c "
import torch
try:
    lens_data = torch.load('\$LENS_FILE', map_location='cpu', weights_only=False)
    print(f'✅ Lens loaded successfully')
    print(f'   Model: {lens_data.get(\"model_name\", \"Unknown\")}')
    print(f'   Layer: {lens_data.get(\"layer_idx\", \"Unknown\")}')
    print(f'   MLP dimension: {lens_data.get(\"mlp_dim\", \"Unknown\")}')
    print(f'   Vocabulary size: {lens_data.get(\"vocab_size\", \"Unknown\")}')
except Exception as e:
    print(f'❌ Error loading lens: {e}')
"
    else
        echo "❌ MLP lens file not found: \$LENS_FILE"
    fi
else
    echo "❌ Training failed with exit code: \$EXIT_CODE"
fi
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

echo "Submitted test job ${JOB_NAME} for layer ${TEST_LAYER}"
echo ""
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs:"
echo "  Output: ${OUTPUT_LOG}"
echo "  Error:  ${ERROR_LOG}"
echo ""
echo "Expected output file:"
echo "  ${OUTPUT_DIR}/EleutherAI_pythia-1.4b/layer_${TEST_LAYER}_mlp_lens.pth"