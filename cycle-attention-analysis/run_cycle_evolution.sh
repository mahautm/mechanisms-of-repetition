#!/bin/bash
#SBATCH --job-name=cycle_evolution_analysis
#SBATCH --qos=alien
#SBATCH --partition=alien
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cycle_evolution_%j.out
#SBATCH --error=logs/cycle_evolution_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🚀 Starting Cycle Evolution Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "================================"

# Change to the source directory
cd /home/mmahaut/projects/parrots/cycle-attention-analysis/src

# Set up environment
export PYTHONPATH="/home/mmahaut/projects/parrots:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE

# Show GPU info
echo "🔧 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo "================================"

# Run the analysis
echo "🔬 Starting attention evolution analysis..."
echo "📊 Processing 300 texts with optimized parameters"
echo "⏱️  Estimated time: 15-30 minutes"
echo ""

# Time the execution
start_time=$(date +%s)

python analyze_cycle_evolution.py

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "================================"
echo "✅ Analysis Complete!"
echo "⏱️  Total time: ${duration} seconds ($(($duration / 60)) minutes)"
echo "📁 Results saved in: ../plots/cycle_evolution/"
echo "📊 Check the plots to see attention evolution patterns!"
echo "Time: $(date)"