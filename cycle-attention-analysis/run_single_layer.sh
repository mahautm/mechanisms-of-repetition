#!/bin/bash
#SBATCH --job-name=attention_analysis_fixed
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Parse arguments
LAYER=$1
MODEL_NAME=$2
N_SAMPLES=$3
N_CYCLES=$4

echo "Starting FIXED attention analysis for layer $LAYER"
echo "Model: $MODEL_NAME"
echo "Samples: $N_SAMPLES" 
echo "Cycles: $N_CYCLES"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Set PYTHONPATH
export PYTHONPATH="/home/mmahaut/projects/parrots:$PYTHONPATH"

# Navigate to project root
cd /home/mmahaut/projects/parrots/cycle-attention-analysis

# Create output directory (remove conflicting files first)
echo "Setting up directory structure..."
if [ -f "data" ]; then rm -f data; fi
if [ -f "results" ]; then rm -f results; fi

mkdir -p data/results/layer_${LAYER}
mkdir -p logs
mkdir -p plots

# Verify directories were created
if [ ! -d "data/results/layer_${LAYER}" ]; then
    echo "ERROR: Failed to create output directory"
    ls -la data/ 2>/dev/null || echo "data directory doesn't exist"
    ls -la data/results/ 2>/dev/null || echo "results directory doesn't exist"
    exit 1
fi

echo "Directory structure ready"

# Run the FIXED analysis
cd src

python main_fixed_v2.py \
    --model-name "$MODEL_NAME" \
    --layer $LAYER \
    --n-cycles $N_CYCLES \
    --n-samples $N_SAMPLES \
    --batch-size 4 \
    --max-length 256 \
    --max-new-tokens 1000 \
    --output-dir "../data/results/layer_${LAYER}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed FIXED analysis for layer $LAYER successfully"
else
    echo "ERROR: FIXED analysis failed for layer $LAYER with exit code $EXIT_CODE"
fi

exit $EXIT_CODE