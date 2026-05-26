#!/bin/bash

# Multi-category alluvial plot generation script for SLURM cluster
# Generates alluvial plots for natural, ICL, and no-cycle ICL data

# SLURM Parameters
MEM="32G"
PARTITION="alien"
QOS="alien"
GRES="gpu:0"  # CPU only for plot generation

# Configuration
LAYER=19  # Change this to your target layer
BASE_PATH="/home/mmahaut/projects/parrots/outputs_multihead_full"
MODEL_NAME="EleutherAI/pythia-1.4b"

# Output directory for plots
OUTPUT_DIR="${BASE_PATH}/alluvial_plots_multi_category"

echo "Starting multi-category alluvial plot generation"
echo "Layer: ${LAYER}"
echo "Data path: ${BASE_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

JOB_NAME="alluvial_multicategory_L${LAYER}"
ERROR_LOG="${OUTPUT_DIR}/alluvial_generation.err"
OUTPUT_LOG="${OUTPUT_DIR}/alluvial_generation.out"

echo "Submitting alluvial plot generation job..."

# Create the plot generation script
script=$(cat <<EOF
echo "SLURM Job Parameters:" > ${OUTPUT_DIR}/sbatch_params.txt
echo "SBATCH_JOB_ID=\$SLURM_JOB_ID" >> ${OUTPUT_DIR}/sbatch_params.txt
echo "SBATCH_JOB_NAME=\$SLURM_JOB_NAME" >> ${OUTPUT_DIR}/sbatch_params.txt
echo "SLURMD_NODENAME=\$SLURMD_NODENAME" >> ${OUTPUT_DIR}/sbatch_params.txt

source ~/.bashrc
echo "Running on node: \$SLURMD_NODENAME"

# Activate environment
conda activate parr

# Change to project directory
cd ~/projects/parrots/

echo "Generating alluvial plots for layer ${LAYER}"
echo "Categories: Natural, ICL, No-cycle ICL"

# Run the alluvial plot generation script
python run_alluvial_only.py

# Move generated plots to output directory
mv alluvial_layer_${LAYER}_*.png "${OUTPUT_DIR}/" 2>/dev/null || true
mv alluvial_layer_${LAYER}_*.pdf "${OUTPUT_DIR}/" 2>/dev/null || true

echo "Alluvial plot generation completed!"
echo "Results saved to: ${OUTPUT_DIR}"

# List generated files
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}"/alluvial_layer_${LAYER}_*
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
    --nice=10 \
    --wrap="$script"

echo "Submitted ${JOB_NAME}"
echo ""
echo "This will generate alluvial plots for:"
echo "  📊 Natural repetition patterns"
echo "  📊 ICL repetition patterns" 
echo "  📊 No-cycle ICL repetition patterns"
echo ""
echo "Monitor with: squeue -u \$USER | grep alluvial"
echo "Results will be in: ${OUTPUT_DIR}"
echo ""
echo "Expected output files:"
echo "  ├── alluvial_layer_${LAYER}_natural_paper.png/.pdf"
echo "  ├── alluvial_layer_${LAYER}_icl_paper.png/.pdf"
echo "  └── alluvial_layer_${LAYER}_no_cycle_icl_paper.png/.pdf"