#!/bin/bash

# Attention ablation analysis with natural-to-no-cycle-ICL ordering
# Run this on SLURM cluster to analyze attention head importance across checkpoints

# SLURM Parameters
MEM="100G"
PARTITION="alien"
QOS="alien"
GRES="gpu:1"  # Need GPU for model inference
EXCLUDE="node044,node043"

# Configuration
MODEL_NAME="EleutherAI/pythia-1.4b"
BASE_PATH="/home/mmahaut/projects/parrots/outputs_multihead_full"
BENCHMARK="hellaswag"  # Options: hellaswag, copa, arc
N_SAMPLES=100  # Number of samples for testing (reduce for faster runs)

# Analysis type - choose one:
ANALYSIS_TYPE="across_checkpoints"  # Options: "single_checkpoint", "across_checkpoints", "plot_only"

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/ablation_results_natural_to_no_cycle_icl"

echo "Starting attention ablation analysis with natural-to-no-cycle-ICL ordering"
echo "Model: ${MODEL_NAME}"
echo "Base path: ${BASE_PATH}"
echo "Benchmark: ${BENCHMARK}"
echo "Analysis type: ${ANALYSIS_TYPE}"
echo "Samples: ${N_SAMPLES}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

JOB_NAME="ablation_nat_to_nocycle_${BENCHMARK}"
ERROR_LOG="${OUTPUT_DIR}/ablation_analysis.err"
OUTPUT_LOG="${OUTPUT_DIR}/ablation_analysis.out"

echo "Submitting ablation analysis job..."

# Create the analysis script
script=$(cat <<EOF
echo "SLURM Job Parameters:" > ${OUTPUT_DIR}/sbatch_params.txt
echo "SBATCH_JOB_ID=\$SLURM_JOB_ID" >> ${OUTPUT_DIR}/sbatch_params.txt
echo "SBATCH_JOB_NAME=\$SLURM_JOB_NAME" >> ${OUTPUT_DIR}/sbatch_params.txt
echo "SLURMD_NODENAME=\$SLURMD_NODENAME" >> ${OUTPUT_DIR}/sbatch_params.txt

source ~/.bashrc
echo "Running on node: \$SLURMD_NODENAME"

# Activate environment
conda activate parr
module load CUDA/12.2.0

# Change to project directory
cd ~/projects/parrots/

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running attention ablation analysis with natural-to-no-cycle-ICL ordering"
echo "This will analyze head importance from most natural-like to most no-cycle-ICL-like"

# Run the appropriate analysis based on type
if [ "${ANALYSIS_TYPE}" = "single_checkpoint" ]; then
    echo "Running single checkpoint analysis (latest model)..."
    
    python parrots/aa_fortu/attention_ablation.py \\
        --run_experiments \\
        --benchmark ${BENCHMARK} \\
        --n_samples ${N_SAMPLES} \\
        --model_name "${MODEL_NAME}" \\
        --base_path "${BASE_PATH}"
        
elif [ "${ANALYSIS_TYPE}" = "across_checkpoints" ]; then
    echo "Running comprehensive analysis across all checkpoints..."
    
    python parrots/aa_fortu/attention_ablation.py \\
        --run_across_checkpoints \\
        --benchmark ${BENCHMARK} \\
        --n_samples ${N_SAMPLES} \\
        --model_name "${MODEL_NAME}" \\
        --base_path "${BASE_PATH}" \\
        --checkpoints step1 step1000 step5000 step7000 step10000 step100000 steplatest
        
elif [ "${ANALYSIS_TYPE}" = "plot_only" ]; then
    echo "Plotting existing results..."
    
    python parrots/aa_fortu/attention_ablation.py \\
        --plot_results \\
        --benchmark ${BENCHMARK}
        
else
    echo "Unknown analysis type: ${ANALYSIS_TYPE}"
    exit 1
fi

# Move results to output directory
mv ablation_results_${BENCHMARK}*.json "${OUTPUT_DIR}/" 2>/dev/null || true
mv ablation_results_${BENCHMARK}*.png "${OUTPUT_DIR}/" 2>/dev/null || true

echo "Ablation analysis completed!"
echo "Results saved to: ${OUTPUT_DIR}"

# List generated files
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}"/ablation_results_${BENCHMARK}*

echo ""
echo "Analysis Summary:"
echo "=================="
echo "Head ordering: Natural-like (low scores) → No-cycle ICL-like (high scores)"
echo "Ablation strategies:"
echo "  1. Natural-first: Remove most natural-like heads first"
echo "  2. No-cycle ICL-first: Remove most no-cycle ICL-like heads first"
echo ""
echo "Metrics tracked:"
echo "  - Repetition rate (natural text generation)"
echo "  - ICL repetition rate (in-context learning)"
echo "  - LAMBADA accuracy (word prediction)"
echo "  - ${BENCHMARK} accuracy (reasoning benchmark)"
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
echo ""
echo "This will run attention ablation analysis with the new ordering:"
echo "  🧠 Natural-like heads (negative composite scores)"
echo "  ➡️  Intermediate heads"
echo "  🔄 No-cycle ICL-like heads (positive composite scores)"
echo ""
echo "Ablation experiments:"
echo "  📊 Natural-first: Tests impact of removing natural text processing"
echo "  📊 No-cycle ICL-first: Tests impact of removing repetition mechanisms"
echo ""
echo "Monitor with: squeue -u \$USER | grep ablation"
echo "Results will be in: ${OUTPUT_DIR}"
echo ""
echo "Expected output files:"
echo "  ├── ablation_results_${BENCHMARK}_all_checkpoints.json (comprehensive results)"
echo "  ├── ablation_results_${BENCHMARK}_all_checkpoints.png (plots)"
echo "  ├── ablation_results_${BENCHMARK}_natural_first.json"
echo "  ├── ablation_results_${BENCHMARK}_no_cycle_icl_first.json"
echo "  └── sbatch_params.txt (job details)"