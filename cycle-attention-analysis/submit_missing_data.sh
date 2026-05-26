#!/bin/bash
# Submit sbatch jobs to regenerate missing no_cycle_icl and no_cycle data for layers 5 and 9

echo "🔄 Submitting jobs for missing sequence types (layers 5 and 9)"
echo ""

cd /home/mmahaut/projects/parrots/cycle-attention-analysis

# Layer 5 - no_cycle_icl
echo "📊 Submitting Layer 5 - no_cycle_icl"
JOB1=$(sbatch run_parametric_fixed.sh 4 5 no_cycle_icl steplatest | awk '{print $4}')
echo "   Job ID: $JOB1"

# Layer 5 - no_cycle  
echo "📊 Submitting Layer 5 - no_cycle"
JOB2=$(sbatch run_parametric_fixed.sh 4 5 no_cycle steplatest | awk '{print $4}')
echo "   Job ID: $JOB2"

# Layer 9 - no_cycle_icl
echo "📊 Submitting Layer 9 - no_cycle_icl"
JOB3=$(sbatch run_parametric_fixed.sh 4 9 no_cycle_icl steplatest | awk '{print $4}')
echo "   Job ID: $JOB3"

# Layer 9 - no_cycle
echo "📊 Submitting Layer 9 - no_cycle"
JOB4=$(sbatch run_parametric_fixed.sh 4 9 no_cycle steplatest | awk '{print $4}')
echo "   Job ID: $JOB4"

echo ""
echo "✅ All 4 jobs submitted!"
echo "💡 Monitor with: squeue -u $USER | grep -E \"($JOB1|$JOB2|$JOB3|$JOB4)\""
echo "📁 Check logs in: logs/cycle_evo_c4_l[59]_*.{out,err}"
echo "⏳ Each job will take ~10-30 minutes"
