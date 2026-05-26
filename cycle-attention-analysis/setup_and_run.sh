#!/bin/bash

echo "Setting up dataset cache and running analysis..."

# Navigate to project root
cd /home/mmahaut/projects/parrots/cycle-attention-analysis

# Setup directories
# ./setup_directories.sh

# Download and cache dataset once
echo "Downloading dataset to cache..."
cd src
python download_dataset.py

if [ $? -eq 0 ]; then
    echo "Dataset cached successfully!"
    cd ..
    
    # Cancel any running jobs
    echo "Canceling existing jobs..."
    scancel -u $USER -n attention_analysis
    sleep 5
    
    # Start all layer jobs
    echo "Starting all layer analysis jobs..."
    ./run_all_layers.sh
    
    echo "All jobs submitted! Monitor with: squeue -u $USER"
else
    echo "Failed to cache dataset. Please check your connection and try again."
    exit 1
fi