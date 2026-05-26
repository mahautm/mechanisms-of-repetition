#!/usr/bin/env python3
"""
Script to run MLP evolution analysis and plotting
"""

import sys
import os

# Add the parrots directory to the path
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu')

from multihead_analysis_graphs import (
    load_mlp_results_across_cycles,
    plot_mlp_evolution_by_checkpoint
)

def main():
    """Run MLP evolution analysis"""
    
    # Configuration for MLP pipeline test results
    base_path = "/home/mmahaut/projects/parrots/test_mlp_pipeline_output"
    model_name = "EleutherAI/pythia-1.4b"
    checkpoints = ["step1", "step1000", "step7000", "step10000", "step100000", "steplatest"]
    
    print("Loading MLP results across checkpoints...")
    try:
        results_across_cycles = load_mlp_results_across_cycles(
            base_path=base_path,
            model_name=model_name,
            checkpoints=checkpoints
        )
        
        print(f"Loaded data for checkpoints: {list(results_across_cycles['icl'].keys())}")
        
        # Print summary of data
        for checkpoint in results_across_cycles['icl']:
            natural_layers = len(results_across_cycles['natural'].get(checkpoint, {}))
            icl_layers = len(results_across_cycles['icl'].get(checkpoint, {}))
            print(f"{checkpoint}: {natural_layers} natural layers, {icl_layers} ICL layers")
        
        print("Creating MLP evolution plots...")
        plot_mlp_evolution_by_checkpoint(
            results_across_cycles, 
            save_path="/home/mmahaut/projects/parrots/mlp_evolution_no_step7000.png"
        )
        
        print("MLP analysis complete!")
        
    except Exception as e:
        print(f"Error during MLP analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()