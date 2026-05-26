#!/bin/bash

# Comprehensive multi-head analysis across all layers
# Run this after training is complete to analyze all layers

# SLURM Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044,node043"

# Model Configuration
MODEL_NAME="EleutherAI/pythia-1.4b"
MAX_LAYERS=1
LENS_PATH="/home/mmahaut/projects/parrots/lenses_multihead"

# Analysis Parameters - Optimized for comprehensive sweep
N_CYCLES=4
MAX_LENGTH=32
BATCH_SIZE=8
N_SAMPLES=100  # Reduced for faster sweep
MAX_NEW_TOKENS=1000

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/outputs_multihead_full"

echo "Starting FULL multi-head contrast analysis for ${MODEL_NAME}"
echo "Analyzing ALL ${MAX_LAYERS} layers with multi-head lenses"
echo "Using lenses from: ${LENS_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

CLEAN_MODEL_NAME=$(echo ${MODEL_NAME} | tr '/' '_')

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Submit jobs for all layers
# checkpoint_list=("step1" "step1000" "step5000" "step7000" "step10000" "step100000" "steplatest")
checkpoint_list=("step0")
# checkpoint_list=(false)
for checkpoint in "${checkpoint_list[@]}"; do
    for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
        
        JOB_NAME="multihead_full_${checkpoint}_L${layer_idx}"
        
        # Create specific output directory
        SPECIFIC_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}/${checkpoint}/layer_${layer_idx}"
        mkdir -p ${SPECIFIC_OUTPUT}
        
        ERROR_LOG="${SPECIFIC_OUTPUT}/full_analysis_cyc${N_CYCLES}_ml${MAX_LENGTH}.err"
        OUTPUT_LOG="${SPECIFIC_OUTPUT}/full_analysis_cyc${N_CYCLES}_ml${MAX_LENGTH}.out"

        echo "Submitting full analysis for checkpoint ${checkpoint}, layer ${layer_idx}..."
        
        # Create the analysis script
        script=$(cat <<EOF
echo "SLURM Job Parameters:" > sbatch_params.txt
echo "SBATCH_JOB_ID=\$SLURM_JOB_ID" >> sbatch_params.txt
echo "SBATCH_JOB_NAME=\$SLURM_JOB_NAME" >> sbatch_params.txt
echo "SLURMD_NODENAME=\$SLURMD_NODENAME" >> sbatch_params.txt

source ~/.bashrc
echo "Running on node: \$SLURMD_NODENAME"

# Activate environment
conda activate parr
module load CUDA/12.2.0

# Change to project directory
cd ~/projects/parrots/

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running FULL multi-head analysis for layer ${layer_idx}"
echo "Expected 16 attention heads with individual contrast values"

# Run analysis
poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/ckpt_pipeline_main.py \
    --model-name="${MODEL_NAME}" \
    --single-lens=${layer_idx} \
    --max-layer-idx=${MAX_LAYERS} \
    --lens-path="${LENS_PATH}/${CLEAN_MODEL_NAME}/" \
    --n-cycles=${N_CYCLES} \
    --max-length=${MAX_LENGTH} \
    --max-new-tokens=${MAX_NEW_TOKENS} \
    --batch-size=${BATCH_SIZE} \
    --n-samples=${N_SAMPLES} \
    $([ "${checkpoint}" != "steplatest" ] && echo "--revision=${checkpoint}")

# Move outputs to layer-specific directory
mv *.png "${SPECIFIC_OUTPUT}/" 2>/dev/null || true
mv *.npy "${SPECIFIC_OUTPUT}/" 2>/dev/null || true
mv *.txt "${SPECIFIC_OUTPUT}/" 2>/dev/null || true

echo "Completed full analysis for layer ${layer_idx}"
echo "Results in: ${SPECIFIC_OUTPUT}"

# Print summary of what we found
echo "Expected heatmap entries (16 heads):"
echo "  gpt_neox.layers.${layer_idx}_head_0 through gpt_neox.layers.${layer_idx}_head_15"
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
    
    echo "Submitted ${JOB_NAME}"
    
    # Brief pause between submissions
    # sleep 1
    done
done
echo ""
echo "Submitted ${MAX_LAYERS} full analysis jobs!"
echo ""
echo "This will analyze:"
echo "  - All ${MAX_LAYERS} layers"
echo "  - 16 attention heads per layer"
echo "  - ${N_CYCLES} cycle repetitions"
echo "  - Total expected head analyses: $((MAX_LAYERS * 16))"
echo ""
echo "Monitor with: squeue -u \$USER | grep multihead_full"
echo "Results structure:"
echo "  ${OUTPUT_DIR}/${MODEL_NAME}/"
echo "    ├── layer_0/full_analysis.{out,err} + heatmaps"
echo "    ├── layer_1/full_analysis.{out,err} + heatmaps"
echo "    ├── ..."
echo "    └── layer_23/full_analysis.{out,err} + heatmaps"