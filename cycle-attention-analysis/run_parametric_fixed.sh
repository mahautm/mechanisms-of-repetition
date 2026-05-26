#!/bin/bash
#SBATCH --job-name=cycle_evo_c%1_l%2
#SBATCH --qos=alien
#SBATCH --partition=alien
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/cycle_evo_c%1_l%2_%j.out
#SBATCH --error=logs/cycle_evo_c%1_l%2_%j.err

# Parameters: $1 = cycles, $2 = layer, $3 = seq_type (optional)
CYCLES=${1:-3}
LAYER=${2:-10}
SEQ_TYPE=${3:-all}
CHECKPOINT=${4:-"steplatest"}

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🚀 Parametric Cycle Evolution Analysis"
echo "Parameters: cycles=$CYCLES, layer=$LAYER, seq_type=$SEQ_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "================================"

# Change to the source directory (where modules/ is located)
cd /home/mmahaut/projects/parrots/cycle-attention-analysis/src

# Set up environment
export PYTHONPATH="/home/mmahaut/projects/parrots:$PYTHONPATH"

# Try to set up CUDA environment
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Show CUDA environment
echo "🔧 CUDA Environment:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Show GPU info
echo "🔧 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo "================================"

# Run the parametric analysis
echo "🔬 Starting parametric analysis..."
echo "📊 Cycles: $CYCLES, Layer: $LAYER, Seq type: $SEQ_TYPE"
echo ""

# Time the execution
start_time=$(date +%s)

# Run from src/ but call the script in experiments/analysis_scripts/
python ../experiments/analysis_scripts/analyze_cycle_evolution_parametric.py \
    --cycles $CYCLES \
    --layer $LAYER \
    --seq_type $SEQ_TYPE \
    --all_heads \
    $([ "$CHECKPOINT" != "steplatest" ] && echo "--revision $CHECKPOINT" || echo "")

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "================================"
echo "✅ Parametric Analysis Complete!"
echo "⏱️  Total time: ${duration} seconds ($(($duration / 60)) minutes)"
echo "📁 Results saved in: ../plots/cycle_evolution_parametric/cycles_${CYCLES}/${CHECKPOINT}/"
echo "📊 Parameters: cycles=$CYCLES, layer=$LAYER, seq_type=$SEQ_TYPE"
echo "Time: $(date)"
