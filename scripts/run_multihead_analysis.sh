#!/bin/bash

# Run contrast analysis with multi-head lenses
# This script analyzes attention head contrasts for different cycle counts and layers

# SLURM Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044,node043,node042,node041"

# Model Configuration
MODEL_NAME="EleutherAI/pythia-1.4b"
MAX_LAYERS=24
LENS_PATH="/home/mmahaut/projects/parrots/lenses_multihead"

# Analysis Parameters
MAX_CYCLES=2
CYCLE_LIST=(0 1 2)
LAYER_LIST=(1 5 10 15 20 23)  # Key layers to analyze
SEQUENCE_LENGTHS=(128 256 512)
BATCH_SIZE=10
N_SAMPLES=1000
MAX_NEW_TOKENS=1000

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/outputs_multihead"

echo "Starting multi-head contrast analysis for ${MODEL_NAME}"
echo "Using lenses from: ${LENS_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Function to submit a single analysis job
submit_analysis_job() {
    local layer_idx=$1
    local n_cycles=$2
    local max_length=$3
    
    JOB_NAME="multihead_contrast_L${layer_idx}_C${n_cycles}_S${max_length}"
    
    # Create specific output directory for this configuration
    SPECIFIC_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}/layer_${layer_idx}/cycles_${n_cycles}/length_${max_length}"
    mkdir -p ${SPECIFIC_OUTPUT}
    
    ERROR_LOG="${SPECIFIC_OUTPUT}/analysis.err"
    OUTPUT_LOG="${SPECIFIC_OUTPUT}/analysis.out"
    
    echo "Submitting: Layer ${layer_idx}, Cycles ${n_cycles}, Length ${max_length}"
    
    # Create the analysis script
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

# Check environment
which python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Change to project directory
cd ~/projects/parrots/

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running multi-head contrast analysis"
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Layer: ${layer_idx}"
echo "  Cycles: ${n_cycles}"
echo "  Max Length: ${max_length}"
echo "  Lens Path: ${LENS_PATH}/${MODEL_NAME}/"
echo "  Output Dir: ${SPECIFIC_OUTPUT}"

# Run contrast analysis
poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/ckpt_pipeline_main.py \
    --model-name="${MODEL_NAME}" \
    --single-lens=${layer_idx} \
    --max-layer-idx=${MAX_LAYERS} \
    --lens-path="${LENS_PATH}/${MODEL_NAME}/" \
    --n-cycles=${n_cycles} \
    --max-length=${max_length} \
    --max-new-tokens=${MAX_NEW_TOKENS} \
    --batch-size=${BATCH_SIZE} \
    --n-samples=${N_SAMPLES} \
    --no-head-analysis=false

# Move output files to specific directory
mv *.png "${SPECIFIC_OUTPUT}/" 2>/dev/null || echo "No PNG files to move"
mv *.npy "${SPECIFIC_OUTPUT}/" 2>/dev/null || echo "No NPY files to move"
mv *.txt "${SPECIFIC_OUTPUT}/" 2>/dev/null || echo "No TXT files to move"

echo "Analysis completed for Layer ${layer_idx}, Cycles ${n_cycles}, Length ${max_length}"
echo "Results saved to: ${SPECIFIC_OUTPUT}"
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
}

# Main execution loop
echo "Submitting contrast analysis jobs..."

job_count=0
for max_length in "${SEQUENCE_LENGTHS[@]}"; do
    for n_cycles in "${CYCLE_LIST[@]}"; do
        for layer_idx in "${LAYER_LIST[@]}"; do
            
            submit_analysis_job ${layer_idx} ${n_cycles} ${max_length}
            job_count=$((job_count + 1))
            
            # Small delay to avoid overwhelming the scheduler
            sleep 1
            
            # Pause every 10 jobs to avoid too many concurrent submissions
            if (( job_count % 10 == 0 )); then
                echo "Submitted ${job_count} jobs, pausing briefly..."
                sleep 5
            fi
        done
    done
done

echo ""
echo "All ${job_count} analysis jobs submitted!"
echo ""
echo "Job configuration summary:"
echo "  Layers analyzed: ${LAYER_LIST[*]}"
echo "  Cycle counts: ${CYCLE_LIST[*]}"
echo "  Sequence lengths: ${SEQUENCE_LENGTHS[*]}"
echo "  Total jobs: ${job_count}"
echo ""
echo "Monitor progress with: squeue -u \$USER"
echo "Check results in: ${OUTPUT_DIR}/${MODEL_NAME}/"
echo ""
echo "Expected analysis structure:"
echo "  ${OUTPUT_DIR}/${MODEL_NAME}/layer_X/cycles_Y/length_Z/"
echo "    ├── analysis.out"
echo "    ├── analysis.err"
echo "    ├── unexpected_lens_heatmap_contrast_Y.png"
echo "    └── [other analysis files]"

# Optional: Submit a summary job that runs after all analysis is complete
echo ""
echo "To run a comprehensive analysis across all layers after training:"
echo "  bash scripts/run_full_multihead_analysis.sh"