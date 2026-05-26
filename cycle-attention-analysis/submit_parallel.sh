#!/bin/bash

# Parallel batch submission for cycle evolution analysis
# This will launch multiple SLURM jobs in parallel for different cycle/layer combinations

echo "🚀 Launching Parallel Cycle Evolution Analysis"
echo "This will submit multiple SLURM jobs for different cycle/layer combinations"
echo ""

# Make scripts executable
chmod +x run_parametric.sh

# Define parameter combinations to test - optimized for speed and valid parameters
CYCLES_TO_TEST=(4)
LAYERS_TO_TEST=($(seq 0 23))
CHECKPOINTS_TO_TEST=("step1" "step1000" "step10000" "step100000" "steplatest")

echo "📊 Parameter combinations to test:"
echo "Cycles: ${CYCLES_TO_TEST[@]}"
echo "Layers: ${LAYERS_TO_TEST[@]}"
echo "Total jobs: $((${#CYCLES_TO_TEST[@]} * ${#LAYERS_TO_TEST[@]} * ${#CHECKPOINTS_TO_TEST[@]}))"
echo ""

# Array to store job IDs
job_ids=()

# Submit jobs for all combinations
for cycles in "${CYCLES_TO_TEST[@]}"; do
    for layer in "${LAYERS_TO_TEST[@]}"; do
        for checkpoint in "${CHECKPOINTS_TO_TEST[@]}"; do

            echo "📤 Submitting job: cycles=$cycles, layer=$layer, checkpoint=$checkpoint"

            # Submit the job and capture job ID
            job_id=$(sbatch --parsable run_parametric.sh $cycles $layer all $checkpoint)
            job_ids+=($job_id)
            
            echo "   ✅ Submitted job $job_id"
            
            # Small delay to avoid overwhelming the scheduler
            # sleep 0.5
        done
    done
done

echo ""
echo "🎉 All jobs submitted!"
echo "📊 Total jobs: ${#job_ids[@]}"
echo "📋 Job IDs: ${job_ids[@]}"
echo ""

echo "📊 Monitor progress with:"
echo "   squeue -u $USER"
echo "   watch -n 5 'squeue -u $USER'"
echo ""

echo "📝 View specific job logs with:"
echo "   tail -f logs/cycle_evo_c[CYCLES]_l[LAYER]_[JOB_ID].out"
echo ""

echo "⏹️  Cancel all jobs if needed:"
echo "   scancel ${job_ids[@]}"
echo ""

echo "📁 Results will be organized in:"
echo "   plots/cycle_evolution_parametric/cycles_[C]/layer_[L]/"
echo ""

echo "⏱️  Expected completion time: 10-20 minutes per job"
echo "🔄 Jobs will run in parallel, so total time should be ~20-30 minutes"

# Save job IDs for later reference
echo "${job_ids[@]}" > job_ids.txt
echo "💾 Job IDs saved to job_ids.txt"