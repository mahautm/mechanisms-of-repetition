#!/bin/bash

echo "Starting entropy analysis of output logit distributions..."
echo "Node: $(hostname)"
echo "Date: $(date)"

cd /home/mmahaut/projects/parrots

# Run entropy analysis with srun using alien QOS and partition, conda environment parr
srun --qos=alien --partition=alien --time=01:00:00 --mem=160G --cpus-per-task=4 --gres=gpu:1 \
    /home/mmahaut/.conda/envs/parr/bin/python run_mlp_evolution.py

echo "Entropy analysis completed at $(date)"