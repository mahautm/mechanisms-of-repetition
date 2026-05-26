#!/usr/bin/env python3
"""
Run cycle evolution plotting on SLURM
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

from parrots.aa_fortu.multihead_analysis_graphs import (
    load_multihead_results_across_cycles,
    plot_cycle_evolution_by_checkpoint
)

def main():
    print("Loading multihead results across cycles...")
    
    # Load results across cycles
    results_across_cycles = load_multihead_results_across_cycles(
        base_path='/home/mmahaut/projects/parrots/outputs_multihead_full',
        model_name='EleutherAI/pythia-1.4b',
        checkpoints=['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest'],
        cycle_range=[0, 1, 2, 3, 4, 5],
        max_length=32
    )
    
    print("Creating horizontal cycle evolution plot...")
    
    # Create the horizontal cycle evolution plot
    plot_cycle_evolution_by_checkpoint(
        results_across_cycles,
        save_path='/home/mmahaut/projects/parrots/cycle_evolution_horizontal.png'
    )
    
    print("✅ Cycle evolution plot completed!")

if __name__ == "__main__":
    main()