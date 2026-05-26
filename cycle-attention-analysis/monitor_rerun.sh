#!/bin/bash
# Monitor the progress of the rerun jobs

echo "🔍 Checking job status..."
echo ""
squeue -u $USER | grep "python3" | head -5
echo ""

echo "📊 Checking log files..."
echo ""

for logfile in /home/mmahaut/projects/parrots/cycle-attention-analysis/logs/rerun_l*.out; do
    if [ -f "$logfile" ]; then
        echo "=== $(basename $logfile) ==="
        tail -5 "$logfile"
        echo ""
    fi
done

echo "❌ Checking for errors..."
echo ""

for errfile in /home/mmahaut/projects/parrots/cycle-attention-analysis/logs/rerun_l*.err; do
    if [ -f "$errfile" ] && [ -s "$errfile" ]; then
        echo "=== $(basename $errfile) ==="
        tail -10 "$errfile"
        echo ""
    fi
done

echo "✅ To check if data files were updated:"
echo "   ls -lht /home/mmahaut/projects/parrots/cycle-attention-analysis/data/cycle_evolution_parametric/cycles_4/steplatest/*_c4_l[59]_* | head"
