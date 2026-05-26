#!/bin/bash
#SBATCH --job-name=olmo_attn_fallback
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/olmo_attention_fallback_%j.out
#SBATCH --error=logs/olmo_attention_fallback_%j.err

echo "🚀 Starting OLMo-1B Attention Fallback Analysis"
echo "================================================"
date

# Model configuration
MODEL_NAME="allenai/OLMo-1B-hf"
N_SAMPLES=100
SEED=${1:-42}  # Accept seed as first argument, default to 42
OUTPUT_DIR="/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison_allenai_OLMo-1B-hf_seed${SEED}"

# Create logs directory
mkdir -p /home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/logs

# Activate environment
source ~/.bashrc
conda activate parr

# Change to script directory
cd /home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality

echo ""
echo "📊 Step 1: Generate attention fallback comparison data"
echo "Model: $MODEL_NAME"
echo "Samples: $N_SAMPLES per sequence type"
echo "Output: $OUTPUT_DIR"
echo ""

python compare_attention_fallback_natural_vs_nocycle.py \
    --model_name "$MODEL_NAME" \
    --n_samples $N_SAMPLES \
    --output_dir "$OUTPUT_DIR" \
    --seed $SEED \
    --batch_size 64

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Attention fallback data generated successfully"
    echo ""
    echo "📊 Step 2: Create publication-ready comparison figure"
    echo ""
    
    python create_paper_comparison_figure.py \
        --model_name "$MODEL_NAME" \
        --results_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ All figures generated successfully!"
        echo ""
        echo "📁 Results saved to: $OUTPUT_DIR"
        echo "   📊 paper_figure_natural_vs_nocycle_comparison.png"
        echo "   📊 paper_figure_natural_vs_icl_clean_allenai_OLMo-1B-hf.png"
        echo "   📝 attention_fallback_natural_vs_nocycle_report.md"
        echo "   💾 attention_fallback_comparison_results.json"
    else
        echo ""
        echo "❌ Figure generation failed"
        exit 1
    fi
else
    echo ""
    echo "❌ Attention fallback analysis failed"
    exit 1
fi

echo ""
echo "🎉 Complete!"
date
