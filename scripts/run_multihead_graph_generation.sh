#!/bin/bash

# Multi-head analysis graph generation with 4-category support
# Run this after multihead analysis is complete to generate all updated graphs

# SLURM Parameters
MEM="50G"
PARTITION="alien"
GRES="gpu:0"  # CPU only for graph generation
QOS="alien"

# Model Configuration
MODEL_NAME="EleutherAI/pythia-1.4b"
BASE_PATH="/home/mmahaut/projects/parrots/outputs_multihead_full"

# Output directory for graphs
OUTPUT_DIR="${BASE_PATH}/graphs_with_no_cycle_icl"

echo "Starting multi-head graph generation with 4-category support"
echo "Model: ${MODEL_NAME}"
echo "Data path: ${BASE_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

JOB_NAME="multihead_graphs_4cat"
ERROR_LOG="${OUTPUT_DIR}/graph_generation.err"
OUTPUT_LOG="${OUTPUT_DIR}/graph_generation.out"

echo "Submitting graph generation job..."

# Create the graph generation script
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

echo "Generating updated multihead analysis graphs with 4-category support"
echo "This includes: natural, icl, successful_icl, and no_cycle_icl categories"

# Run the multihead analysis graphs script directly
python parrots/aa_fortu/multihead_analysis_graphs.py \
    --base_path "${BASE_PATH}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}"

echo "Graph generation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
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
echo "This will generate updated graphs with 4-category support:"
echo "  - Natural contrasts"
echo "  - ICL contrasts" 
echo "  - Successful ICL contrasts"
echo "  - No-cycle ICL contrasts (NEW)"
echo ""
echo "Monitor with: squeue -u \$USER | grep multihead_graphs"
echo "Results will be in: ${OUTPUT_DIR}"
echo ""
echo "Expected output files:"
echo "  ├── multihead_checkpoint_evolution_4cat.png"
echo "  ├── multihead_cycle_evolution_4cat.png"
echo "  ├── multihead_cycle_summary_4cat.png"
echo "  ├── multihead_heatmap_step1000_4cat.png"
echo "  ├── multihead_heatmap_step10000_4cat.png"
echo "  ├── multihead_heatmap_step100000_4cat.png"
echo "  ├── multihead_heatmap_steplatest_4cat.png"
echo "  └── repetition_alluvial_plot_4cat.png"