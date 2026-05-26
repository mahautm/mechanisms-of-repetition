#!/bin/bash
#SBATCH --job-name=attention_fallback_layer_%a
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --array=0-23
#SBATCH --output=logs/attention_fallback_layer_%a.out
#SBATCH --error=logs/attention_fallback_layer_%a.err

echo "🚀 Starting attention fallback analysis for layer $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID" 
echo "Node: $HOSTNAME"
echo "Started at: $(date)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate parr

# Set layer from array task ID
TARGET_LAYER=$SLURM_ARRAY_TASK_ID

echo "📊 Analyzing layer: $TARGET_LAYER"

# Run the analysis for this specific layer
cd /home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality

python compare_attention_fallback_natural_vs_nocycle.py \
    --n_samples 25 \
    --target_layer $TARGET_LAYER \
    --output_dir ./plots/attention_fallback_per_layer/layer_${TARGET_LAYER}

EXIT_CODE=$?

echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Layer $TARGET_LAYER analysis completed successfully"
else
    echo "❌ Layer $TARGET_LAYER analysis failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE