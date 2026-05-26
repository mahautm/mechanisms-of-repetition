#!/bin/bash

# Complete workflow for parallel cycle evolution analysis

echo "🚀 Parallel Cycle Evolution Analysis Workflow"
echo "=============================================="
echo ""

# Step 1: Submit all parallel jobs
echo "📤 Step 1: Submitting parallel jobs..."
cd /home/mmahaut/projects/parrots/cycle-attention-analysis
./submit_parallel.sh

echo ""
echo "⏳ Jobs submitted! Waiting for completion..."
echo "💡 You can monitor progress with: watch -n 5 'squeue -u $USER'"
echo ""

# Wait for user to confirm jobs are done
read -p "🔄 Press Enter when all jobs are complete (check with 'squeue -u $USER')..."

echo ""
echo "📊 Step 2: Aggregating results..."
cd src
python3 aggregate_parallel_results.py

echo ""
echo "🎉 Complete Workflow Finished!"
echo ""
echo "📁 Check results in:"
echo "  - plots/cycle_evolution_parametric/ (individual results)"
echo "  - plots/aggregated_cycle_evolution/ (combined analysis)"
echo ""
echo "🔍 Key files to examine:"
echo "  📊 parameter_heatmaps.png - Consistency across all combinations"
echo "  📈 cycle_layer_comparison.png - How parameters affect consistency"
echo "  🔬 sequence_type_comparison.png - Natural vs ICL vs No-cycle"
echo ""
echo "❓ This analysis answers your key question:"
echo "   'Does attention move to different tokens in different cycles or is it the same?'"
echo ""
echo "💡 Look for:"  
echo "  - High consistency (>0.7) = Same attention patterns across cycles"
echo "  - Low consistency (<0.3) = Attention shifts between cycles"
echo "  - Medium consistency (0.3-0.7) = Partially consistent patterns"