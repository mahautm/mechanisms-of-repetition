#!/bin/bash
#
# Regenerate prompt vs attention reports for all 24 layers
# with improved token classification including PROGRAMMING category
#

DATA_PATH="../../data/cycle_evolution_parametric/cycles_4/steplatest"
OUTPUT_DIR="../../data/prompt_vs_attention_analysis"
CYCLES=4

echo "🚀 Regenerating prompt vs attention reports for all 24 layers..."
echo "📂 Data path: $DATA_PATH"
echo "📁 Output dir: $OUTPUT_DIR"
echo ""

# Loop through all 24 layers (0-23)
for layer in {0..23}; do
    echo "=================================================="
    echo "📊 Processing Layer $layer..."
    echo "=================================================="
    
    srun --qos=alien --partition=alien --mem=32G \
        python3 analyze_prompt_vs_attention.py \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --layer $layer \
        --cycles $CYCLES \
        2>&1 | grep -E "(Processing|Analyzing|Creating|Summary|Completed|✅|❌)" || true
    
    if [ $? -eq 0 ]; then
        echo "✅ Layer $layer completed successfully"
    else
        echo "❌ Layer $layer failed"
    fi
    echo ""
done

echo ""
echo "=================================================="
echo "✅ All layers processed!"
echo "=================================================="
echo ""
echo "📊 Generating summary statistics..."

python3 create_bias_ratio_summary.py \
    --reports_dir "$OUTPUT_DIR" \
    --output_dir "../../plots/attention_bias_summary"

echo ""
echo "🎉 Complete! All reports and summaries have been regenerated."
