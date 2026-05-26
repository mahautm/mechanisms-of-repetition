#!/bin/bash
# Rerun missing no_cycle_icl and no_cycle data for layers 5 and 9

echo "🔄 Rerunning missing sequence types for layers 5 and 9"
echo "This will generate no_cycle_icl and no_cycle data that's currently missing"
echo ""

cd /home/mmahaut/projects/parrots/cycle-attention-analysis

# Layer 5 - no_cycle_icl
echo "📊 Submitting Layer 5 - no_cycle_icl"
sbatch run_parametric.sh 4 5 no_cycle_icl steplatest

# Layer 5 - no_cycle  
echo "📊 Submitting Layer 5 - no_cycle"
sbatch run_parametric.sh 4 5 no_cycle steplatest

# Layer 9 - no_cycle_icl
echo "📊 Submitting Layer 9 - no_cycle_icl"
sbatch run_parametric.sh 4 9 no_cycle_icl steplatest

# Layer 9 - no_cycle
echo "📊 Submitting Layer 9 - no_cycle"
sbatch run_parametric.sh 4 9 no_cycle steplatest

echo ""
echo "✅ All jobs submitted!"
echo "💡 Monitor with: squeue -u $USER"
echo "📁 Check logs in: logs/cycle_evo_c4_l*"
