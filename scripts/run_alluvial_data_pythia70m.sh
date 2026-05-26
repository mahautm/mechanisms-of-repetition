#!/bin/bash
#SBATCH --job-name=alluvial_pythia70m
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/mmahaut/projects/parrots/logs/alluvial_pythia70m_%j.out
#SBATCH --error=/home/mmahaut/projects/parrots/logs/alluvial_pythia70m_%j.err

# Generate alluvial data for Pythia-70m across all checkpoints
# This runs the same analysis as ckpt_pipeline_main but without needing lenses

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0

cd /home/mmahaut/projects/parrots

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="EleutherAI/pythia-70m"
LAYER=4  # 75% depth for 6-layer model
N_SAMPLES=300
MAX_LENGTH=32
MAX_NEW_TOKENS=1000
N_CYCLES=4
OUTPUT_DIR="./outputs_multihead_full"

echo "Generating alluvial data for ${MODEL}"
echo "Layer: ${LAYER}, Samples: ${N_SAMPLES}"

# Checkpoints to process
CHECKPOINTS=("step1" "step1000" "step5000" "step10000" "step100000" "steplatest")

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing checkpoint: ${checkpoint}"
    echo "========================================"
    
    if [ "$checkpoint" == "steplatest" ]; then
        python3 generate_alluvial_data.py \
            --model-name="${MODEL}" \
            --n-samples=${N_SAMPLES} \
            --max-length=${MAX_LENGTH} \
            --max-new-tokens=${MAX_NEW_TOKENS} \
            --n-cycles=${N_CYCLES} \
            --output-dir="${OUTPUT_DIR}" \
            --layer=${LAYER}
    else
        python3 generate_alluvial_data.py \
            --model-name="${MODEL}" \
            --revision="${checkpoint}" \
            --n-samples=${N_SAMPLES} \
            --max-length=${MAX_LENGTH} \
            --max-new-tokens=${MAX_NEW_TOKENS} \
            --n-cycles=${N_CYCLES} \
            --output-dir="${OUTPUT_DIR}" \
            --layer=${LAYER}
    fi
done

echo ""
echo "All checkpoints processed!"
echo "Output directory: ${OUTPUT_DIR}/${MODEL}/"
