#!/bin/bash

# Test the MLP pipeline after training lens using the working SLURM pattern

# SLURM Parameters
MEM="32G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien" 
EXCLUDE="node044,node043"

# Model Configuration
MODEL_NAME="EleutherAI/pythia-1.4b"
MAX_LAYERS=24
# CHECKPOINTS=("steplatest")
CHECKPOINTS=("step0")
# CHECKPOINTS=("step1" "step1000" "step5000" "step7000" "step10000" "step100000" "steplatest")
N_CYCLES=2
MAX_LENGTH=32
BATCH_SIZE=10
N_SAMPLES=1000
MAX_NEW_TOKENS=1000

# Pipeline test
LENS_DIR="/home/mmahaut/projects/parrots/lenses_mlp/${MODEL_NAME//\//_}"
OUTPUT_DIR="/home/mmahaut/projects/parrots/test_mlp_pipeline_output"

echo "Testing MLP pipeline for ${MODEL_NAME} - all ${MAX_LAYERS} layers"
echo "Lens directory: ${LENS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Test pipeline for each layer
for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
    for CHECKPOINT in "${CHECKPOINTS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Testing layer ${layer_idx}"
        echo "=========================================="
        # Check if lens file exists
        LENS_FILE="${LENS_DIR}/layer_${layer_idx}_mlp_lens.pth"
        if [ ! -f "$LENS_FILE" ]; then
            echo "❌ MLP lens file not found: $LENS_FILE"
            echo "Skipping layer ${layer_idx} - lens not trained yet"
            continue
        fi

        echo "✅ Found MLP lens file: $LENS_FILE"

        JOB_NAME="test_mlp_pipeline_layer_${layer_idx}"
        LOG_DIR="${OUTPUT_DIR}/${CHECKPOINT}"
        mkdir -p ${LOG_DIR}
        ERROR_LOG="${LOG_DIR}/pipeline_test_layer_${layer_idx}.err"
        OUTPUT_LOG="${LOG_DIR}/pipeline_test_layer_${layer_idx}.out"

        echo "Submitting pipeline test job for layer ${layer_idx}..."

        # Create the pipeline test script
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
cd ~/projects/parrots/parrots/aa_fortu/

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Testing MLP pipeline for layer ${layer_idx}"
echo "Model: ${MODEL_NAME}"
echo "Lens directory: ${LENS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Run pipeline test
if [ "${CHECKPOINT}" = "steplatest" ]; then
    # For steplatest, don't specify revision (uses default/latest)
    poetry run python ckpt_pipeline_main.py \
        --use-mlp \
        --model-name="${MODEL_NAME}" \
        --lens-path="${LENS_DIR}" \
        --single-lens=${layer_idx} \
        --max-layer-idx=${MAX_LAYERS} \
        --n-cycles=${N_CYCLES} \
        --batch-size=${BATCH_SIZE} \
        --max-length=${MAX_LENGTH} \
        --max-new-tokens=${MAX_NEW_TOKENS} \
        --n-samples=${N_SAMPLES} 
else
    # For other checkpoints, use the specified revision
    poetry run python ckpt_pipeline_main.py \
        --use-mlp \
        --model-name="${MODEL_NAME}" \
        --revision="${CHECKPOINT}" \
        --lens-path="${LENS_DIR}" \
        --single-lens=${layer_idx} \
        --max-layer-idx=${MAX_LAYERS} \
        --n-cycles=${N_CYCLES} \
        --batch-size=${BATCH_SIZE} \
        --max-length=${MAX_LENGTH} \
        --max-new-tokens=${MAX_NEW_TOKENS} \
        --n-samples=${N_SAMPLES} 
fi 

EXIT_CODE=\$?

echo "Pipeline test completed with exit code: \$EXIT_CODE"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ MLP pipeline test completed successfully for layer ${layer_idx}!"
    echo "Output directory: ${OUTPUT_DIR}/layer_${layer_idx}"
    
    # List output files
    if [ -d "${OUTPUT_DIR}/layer_${layer_idx}" ]; then
        echo "Generated files:"
        find "${OUTPUT_DIR}/layer_${layer_idx}" -type f -name "*.png" -o -name "*.json" -o -name "*.txt" | head -10
    fi
else
    echo "❌ Pipeline test failed for layer ${layer_idx} with exit code: \$EXIT_CODE"
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

        echo "Submitted pipeline test job ${JOB_NAME} for layer ${layer_idx}"
        
    done
done

echo ""
echo "=========================================="
echo "All pipeline test jobs submitted!"
echo "=========================================="
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: ${OUTPUT_DIR}/logs/"
echo ""
echo "Expected output directories:"
for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
    echo "  ${OUTPUT_DIR}/layer_${layer_idx}/"
done