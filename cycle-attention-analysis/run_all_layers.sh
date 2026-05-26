#!/bin/bash

MODEL_NAME="EleutherAI/pythia-1.4b"
N_SAMPLES=1000
# Cancel any existing jobs
echo "Canceling any existing attention analysis jobs..."
scancel -u $USER -n attention_analysis
scancel -u $USER -n attention_analysis_fixed
sleep 3
for N_CYCLES in {0..4}
do
    echo "Starting FIXED attention analysis for all layers with $N_CYCLES cycles..."
    echo "Model: $MODEL_NAME"
    echo "Samples: $N_SAMPLES"
    echo "Cycles: $N_CYCLES"

    # Submit jobs for all layers
    for layer in {0..23}
    do
        echo "Submitting job for layer $layer with $N_CYCLES cycles..."
        sbatch run_single_layer.sh $layer "$MODEL_NAME" $N_SAMPLES $N_CYCLES
        # sleep 1
    done

    echo "All FIXED analysis jobs submitted for $N_CYCLES cycles!"
done