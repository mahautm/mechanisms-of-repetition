#!/bin/bash

echo "🚀 Launching Attention Fallback Analysis Across All Layers"
echo "=========================================================="

# Create necessary directories
mkdir -p logs
mkdir -p plots/attention_fallback_per_layer
mkdir -p plots/attention_fallback_all_layers_summary

echo "📁 Created necessary directories"

# Make the batch script executable
chmod +x run_attention_fallback_all_layers.sh

echo "🔄 Submitting SLURM array job for all 24 layers..."

# Submit the array job
JOB_ID=$(sbatch run_attention_fallback_all_layers.sh | grep -o '[0-9]*')

if [ ! -z "$JOB_ID" ]; then
    echo "✅ Submitted job array with ID: $JOB_ID"
    echo "   📊 This will analyze layers 0-23 in parallel"
    echo "   ⏱️  Each layer analysis should take ~20-30 minutes"
    echo "   📋 Monitor progress with: squeue -u $USER"
    echo ""
    echo "📝 Log files will be in: logs/attention_fallback_layer_*.{out,err}"
    echo "📊 Results will be in: plots/attention_fallback_per_layer/layer_*/"
    echo ""
    echo "🔍 To check job status:"
    echo "   squeue -u $USER"
    echo ""
    echo "📈 After all jobs complete, run the summary analysis:"
    echo "   python summarize_all_layers_attention_fallback.py"
    echo ""
    echo "⚡ Or use this one-liner to auto-run summary when jobs finish:"
    echo "   while squeue -u $USER | grep -q $JOB_ID; do sleep 30; done; python summarize_all_layers_attention_fallback.py"
else
    echo "❌ Failed to submit job array"
    exit 1
fi