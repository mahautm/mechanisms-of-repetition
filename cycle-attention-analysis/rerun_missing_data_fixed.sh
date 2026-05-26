#!/bin/bash
# Rerun missing no_cycle_icl and no_cycle data for layers 5 and 9
# Using direct srun commands instead of sbatch to avoid path issues

echo "🔄 Rerunning missing sequence types for layers 5 and 9"
echo "This will generate no_cycle_icl and no_cycle data that's currently missing"
echo ""

cd /home/mmahaut/projects/parrots/cycle-attention-analysis/src

export PYTHONPATH="/home/mmahaut/projects/parrots:$PYTHONPATH"

# Layer 5 - no_cycle_icl
echo "📊 Running Layer 5 - no_cycle_icl"
srun --qos=alien --partition=alien --gres=gpu:1 --mem=32G --time=2:00:00 \
  python3 ../experiments/analysis_scripts/analyze_cycle_evolution_parametric.py \
  --cycles 4 --layer 5 --seq_type no_cycle_icl --all_heads \
  > ../logs/rerun_l5_no_cycle_icl.out 2> ../logs/rerun_l5_no_cycle_icl.err &

# Layer 5 - no_cycle  
echo "📊 Running Layer 5 - no_cycle"
srun --qos=alien --partition=alien --gres=gpu:1 --mem=32G --time=2:00:00 \
  python3 ../experiments/analysis_scripts/analyze_cycle_evolution_parametric.py \
  --cycles 4 --layer 5 --seq_type no_cycle --all_heads \
  > ../logs/rerun_l5_no_cycle.out 2> ../logs/rerun_l5_no_cycle.err &

# Layer 9 - no_cycle_icl
echo "📊 Running Layer 9 - no_cycle_icl"
srun --qos=alien --partition=alien --gres=gpu:1 --mem=32G --time=2:00:00 \
  python3 ../experiments/analysis_scripts/analyze_cycle_evolution_parametric.py \
  --cycles 4 --layer 9 --seq_type no_cycle_icl --all_heads \
  > ../logs/rerun_l9_no_cycle_icl.out 2> ../logs/rerun_l9_no_cycle_icl.err &

# Layer 9 - no_cycle
echo "📊 Running Layer 9 - no_cycle"
srun --qos=alien --partition=alien --gres=gpu:1 --mem=32G --time=2:00:00 \
  python3 ../experiments/analysis_scripts/analyze_cycle_evolution_parametric.py \
  --cycles 4 --layer 9 --seq_type no_cycle --all_heads \
  > ../logs/rerun_l9_no_cycle.out 2> ../logs/rerun_l9_no_cycle.err &

echo ""
echo "✅ All jobs started in background!"
echo "💡 Monitor with: squeue -u $USER"
echo "📁 Check logs in: logs/rerun_l*"
echo "⏳ Jobs will take ~10-30 minutes each"
