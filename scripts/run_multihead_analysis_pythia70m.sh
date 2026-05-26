#!/bin/bash

# Run full multihead attention analysis for Pythia-70m
# Requires trained multihead lenses in lenses_multihead/EleutherAI_pythia-70m/
# Generates data for attention head contrast plots (like multihead_cycle_evolution.png)

# SLURM Parameters
MEM="64G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"

# Model Configuration
MODEL_NAME="EleutherAI/pythia-70m"
MAX_LAYERS=6
LENS_PATH="/home/mmahaut/projects/parrots/lenses_multihead"

# Analysis Parameters
N_CYCLES=4
MAX_LENGTH=32
BATCH_SIZE=8
N_SAMPLES=300
MAX_NEW_TOKENS=1000

# Output directory
OUTPUT_DIR="/home/mmahaut/projects/parrots/outputs_multihead_full"

echo "Starting multihead attention analysis for ${MODEL_NAME}"
echo "Analyzing ${MAX_LAYERS} layers with multihead lenses"

CLEAN_MODEL_NAME=$(echo ${MODEL_NAME} | tr '/' '_')

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs

# Checkpoints to process
checkpoint_list=("step1" "step1000" "step5000" "step10000" "step100000" "steplatest")

for checkpoint in "${checkpoint_list[@]}"; do
    for layer_idx in $(seq 0 $((MAX_LAYERS - 1))); do
        
        JOB_NAME="mh_analysis_70m_${checkpoint}_L${layer_idx}"
        
        # Create specific output directory
        SPECIFIC_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}/${checkpoint}/layer_${layer_idx}"
        mkdir -p ${SPECIFIC_OUTPUT}
        
        ERROR_LOG="${OUTPUT_DIR}/logs/analysis_70m_${checkpoint}_L${layer_idx}.err"
        OUTPUT_LOG="${SPECIFIC_OUTPUT}/full_analysis_cyc${N_CYCLES}_ml${MAX_LENGTH}.out"

        echo "Submitting analysis for checkpoint ${checkpoint}, layer ${layer_idx}..."
        
        # Create the analysis script
        script=$(cat <<EOF
source ~/.bashrc
conda activate parr
module load CUDA/12.2.0

cd ~/projects/parrots/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running multihead analysis for ${MODEL_NAME} ${checkpoint} layer ${layer_idx}"

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

# Move outputs
mv *.png "${SPECIFIC_OUTPUT}/" 2>/dev/null || true
mv *.npy "${SPECIFIC_OUTPUT}/" 2>/dev/null || true

echo "Completed analysis for ${checkpoint} layer ${layer_idx}"
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
            --time=2:00:00 \
            --wrap="$script"
        
        sleep 0.5
    done
done

echo ""
echo "All analysis jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Output: ${OUTPUT_DIR}/${MODEL_NAME}/"
