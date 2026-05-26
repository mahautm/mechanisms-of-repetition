#!/bin/bash

# Test MLP dimensions using the same SLURM environment

# SLURM Parameters
MEM="16G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien" 
EXCLUDE="node044"

echo "Testing MLP dimensions for Pythia model"

JOB_NAME="test_mlp_dimensions"
LOG_DIR="/home/mmahaut/projects/parrots/debug_logs"
mkdir -p ${LOG_DIR}
ERROR_LOG="${LOG_DIR}/mlp_dimensions.err"
OUTPUT_LOG="${LOG_DIR}/mlp_dimensions.out"

echo "Submitting dimension test job..."

# Create the test script
script=$(cat <<EOF
source ~/.bashrc
echo "Running on node: \$SLURMD_NODENAME"
echo "GPU devices: \$CUDA_VISIBLE_DEVICES"

# Activate environment
conda activate parr
module load CUDA/12.2.0

# Change to project directory
cd ~/projects/parrots/

echo "Testing MLP dimensions..."
poetry run python test_mlp_dimensions.py
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

echo "Submitted dimension test job ${JOB_NAME}"
echo ""
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs:"
echo "  Output: ${OUTPUT_LOG}"
echo "  Error:  ${ERROR_LOG}"