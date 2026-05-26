#!/bin/bash
#SBATCH --job-name=analysis_2x2
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mmahaut/projects/parrots/logs/analysis_2x2_%j.out
#SBATCH --error=/home/mmahaut/projects/parrots/logs/analysis_2x2_%j.err

# Run 2x2 analysis: (Natural/ICL) x (Original/Acquired)
# across all checkpoints for both Pythia models

source ~/.bashrc
conda activate parr
module load CUDA/12.2.0

cd /home/mmahaut/projects/parrots
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_SAMPLES=300
MAX_LENGTH=32
MAX_NEW_TOKENS=1000
N_CYCLES=4
OUTPUT_DIR="./outputs_2x2_analysis"

mkdir -p ${OUTPUT_DIR}

CHECKPOINTS=("step1" "step1000" "step5000" "step10000" "step100000" "steplatest")

echo "=========================================="
echo "2x2 Analysis: Natural/ICL x Original/Acquired"
echo "=========================================="

# Pythia-1.4b
MODEL="EleutherAI/pythia-1.4b"
EVOLUTION_FILE="./cycle_evolution_results/cycle_evolution_status_EleutherAI_pythia-1.4b.json"

if [ -f "$EVOLUTION_FILE" ]; then
    echo ""
    echo "Processing ${MODEL}..."
    
    for checkpoint in "${CHECKPOINTS[@]}"; do
        echo ""
        echo "--- ${MODEL} @ ${checkpoint} ---"
        
        if [ "$checkpoint" == "steplatest" ]; then
            python3 analyze_2x2_natural_icl_original_acquired.py \
                --model-name="${MODEL}" \
                --evolution-file="${EVOLUTION_FILE}" \
                --n-samples=${N_SAMPLES} \
                --n-cycles=${N_CYCLES} \
                --max-length=${MAX_LENGTH} \
                --max-new-tokens=${MAX_NEW_TOKENS} \
                --output-dir="${OUTPUT_DIR}"
        else
            python3 analyze_2x2_natural_icl_original_acquired.py \
                --model-name="${MODEL}" \
                --revision="${checkpoint}" \
                --evolution-file="${EVOLUTION_FILE}" \
                --n-samples=${N_SAMPLES} \
                --n-cycles=${N_CYCLES} \
                --max-length=${MAX_LENGTH} \
                --max-new-tokens=${MAX_NEW_TOKENS} \
                --output-dir="${OUTPUT_DIR}"
        fi
    done
else
    echo "Evolution file not found for ${MODEL}: ${EVOLUTION_FILE}"
fi

# Pythia-70m
MODEL="EleutherAI/pythia-70m"
EVOLUTION_FILE="./cycle_evolution_results/cycle_evolution_status_EleutherAI_pythia-70m.json"

if [ -f "$EVOLUTION_FILE" ]; then
    echo ""
    echo "Processing ${MODEL}..."
    
    for checkpoint in "${CHECKPOINTS[@]}"; do
        echo ""
        echo "--- ${MODEL} @ ${checkpoint} ---"
        
        if [ "$checkpoint" == "steplatest" ]; then
            python3 analyze_2x2_natural_icl_original_acquired.py \
                --model-name="${MODEL}" \
                --evolution-file="${EVOLUTION_FILE}" \
                --n-samples=${N_SAMPLES} \
                --n-cycles=${N_CYCLES} \
                --max-length=${MAX_LENGTH} \
                --max-new-tokens=${MAX_NEW_TOKENS} \
                --output-dir="${OUTPUT_DIR}"
        else
            python3 analyze_2x2_natural_icl_original_acquired.py \
                --model-name="${MODEL}" \
                --revision="${checkpoint}" \
                --evolution-file="${EVOLUTION_FILE}" \
                --n-samples=${N_SAMPLES} \
                --n-cycles=${N_CYCLES} \
                --max-length=${MAX_LENGTH} \
                --max-new-tokens=${MAX_NEW_TOKENS} \
                --output-dir="${OUTPUT_DIR}"
        fi
    done
else
    echo "Evolution file not found for ${MODEL}: ${EVOLUTION_FILE}"
fi

echo ""
echo "=========================================="
echo "All 2x2 analyses complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
