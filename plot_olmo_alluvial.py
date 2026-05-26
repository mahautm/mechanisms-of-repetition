#!/usr/bin/env python3
"""
Generate Alluvial Plot for OLMo Checkpoint Evolution
Reads log files from olmo_attention jobs and creates visualization
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import interpolate

# Import the beautiful alluvial class
from run_alluvial_dual import BeautifulDualAlluvial

def load_olmo_data_from_logs(log_dir="logs", model_name="allenai/OLMo-1B-hf", 
                              checkpoints=None, layer=12):
    """Load OLMo repetition data from log files"""
    
    if checkpoints is None:
        checkpoints = [
            'step1000-tokens4B',
            'step343000-tokens1438B', 
            'step425000-tokens1781B',
            'step509000-tokens2134B',
            'step593000-tokens2486B',
            'step738020-tokens3094B'
        ]
    
    # Map checkpoints to job IDs (from the logs we saw)
    checkpoint_to_job = {
        'step1000-tokens4B': '1839766',
        'step343000-tokens1438B': '1839781',
        'step425000-tokens1781B': '1839782',
        'step509000-tokens2134B': '1839783',
        'step593000-tokens2486B': '1839784',
        'step738020-tokens3094B': '1839785',
    }
    
    repetition_data = {}
    log_path = Path(log_dir)
    
    for checkpoint in checkpoints:
        job_id = checkpoint_to_job.get(checkpoint)
        if not job_id:
            print(f"No job ID found for checkpoint: {checkpoint}")
            continue
            
        log_file = log_path / f"olmo_attention_{job_id}.out"
        
        if not log_file.exists():
            print(f"Log file not found: {log_file}")
            continue
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract indices using the format we print
            data_index_match = re.search(rf'layer {layer} data index: \[(.*?)\]', content)
            repetition_index_match = re.search(rf'layer {layer} repetition index: \[(.*?)\]', content)
            no_cycle_match = re.search(rf'layer {layer} no-cycle icl index: \[(.*?)\]', content)
            
            if data_index_match and repetition_index_match:
                # Parse the indices
                data_indices_str = data_index_match.group(1)
                repetition_indices_str = repetition_index_match.group(1)
                
                # Convert to lists of integers
                data_indices = [int(x.strip()) for x in data_indices_str.split(',') if x.strip()]
                repetition_indices = [int(x.strip()) for x in repetition_indices_str.split(',') if x.strip()]
                
                # Parse no-cycle indices if available
                no_cycle_indices = []
                if no_cycle_match:
                    no_cycle_str = no_cycle_match.group(1)
                    if no_cycle_str.strip():
                        no_cycle_indices = [int(x.strip()) for x in no_cycle_str.split(',') if x.strip()]
                
                repetition_data[checkpoint] = {
                    'data_indices': data_indices,
                    'repetition_indices': repetition_indices,
                    'no_cycle_indices': no_cycle_indices,
                    'cycle': 0,
                    'layer': layer
                }
                
                print(f"✓ {checkpoint}: {len(data_indices)} samples, {len(repetition_indices)} repetitions, {len(no_cycle_indices)} no-cycle")
            else:
                print(f"✗ {checkpoint}: Could not parse indices from log")
                
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            continue
    
    return repetition_data

def create_olmo_alluvial_plot(output_file="plots/olmo_checkpoint_evolution.png"):
    """Create alluvial plot showing OLMo checkpoint evolution"""
    
    print("\n" + "="*80)
    print("OLMo Checkpoint Evolution - Alluvial Plot Generator")
    print("="*80 + "\n")
    
    # Load data from logs
    print("Loading data from log files...")
    repetition_data = load_olmo_data_from_logs()
    
    if not repetition_data:
        print("❌ No data loaded! Check log files.")
        return
    
    print(f"\n✓ Loaded {len(repetition_data)} checkpoints")
    
    # Initialize the alluvial plotter
    alluvial = BeautifulDualAlluvial()
    
    # Prepare checkpoint order
    checkpoints = [
        'step1000-tokens4B',
        'step343000-tokens1438B',
        'step425000-tokens1781B', 
        'step509000-tokens2134B',
        'step593000-tokens2486B',
        'step738020-tokens3094B'
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Track which samples are repeating at each checkpoint
    all_samples = set()
    for data in repetition_data.values():
        all_samples.update(data['data_indices'])
    
    print(f"\n📊 Tracking {len(all_samples)} total unique samples across checkpoints")
    
    # For each sample, determine when it starts repeating
    sample_categories = {}  # sample_id -> checkpoint where repetition starts
    
    for sample_id in all_samples:
        for checkpoint in checkpoints:
            if checkpoint in repetition_data:
                data = repetition_data[checkpoint]
                # Check if this sample appears in data_indices and has a repetition
                if sample_id in data['data_indices']:
                    idx_pos = data['data_indices'].index(sample_id)
                    if idx_pos < len(data['repetition_indices']) and data['repetition_indices'][idx_pos] > 0:
                        # This sample is repeating at this checkpoint
                        if sample_id not in sample_categories:
                            sample_categories[sample_id] = checkpoint
                            break
    
    # Count samples by category
    category_counts = defaultdict(lambda: defaultdict(int))
    for checkpoint in checkpoints:
        if checkpoint not in repetition_data:
            continue
        
        for sample_id in all_samples:
            if sample_id in sample_categories:
                # Sample repeats starting from sample_categories[sample_id]
                first_repeat_checkpoint = sample_categories[sample_id]
                if checkpoints.index(checkpoint) >= checkpoints.index(first_repeat_checkpoint):
                    category = f"repeating_since_{first_repeat_checkpoint}"
                    category_counts[checkpoint][category] += 1
                else:
                    category_counts[checkpoint]["non_repeating"] += 1
            else:
                category_counts[checkpoint]["non_repeating"] += 1
    
    # Print category distribution
    print("\n📈 Category Distribution by Checkpoint:")
    for checkpoint in checkpoints:
        if checkpoint in category_counts:
            counts = category_counts[checkpoint]
            print(f"\n{checkpoint}:")
            for cat, count in sorted(counts.items()):
                print(f"  {cat}: {count}")
    
    # Create stacked bars
    bar_width = 1.0
    x_positions = np.arange(len(checkpoints))
    
    # Define category order and colors
    categories = ["non_repeating"] + [f"repeating_since_{cp}" for cp in checkpoints]
    colors = {
        "non_repeating": '#D0D0D0',
        f"repeating_since_{checkpoints[0]}": '#4A90E2',
        f"repeating_since_{checkpoints[1]}": '#9B59B6',
        f"repeating_since_{checkpoints[2]}": '#E67E22',
        f"repeating_since_{checkpoints[3]}": '#E74C3C',
        f"repeating_since_{checkpoints[4]}": '#27AE60',
        f"repeating_since_{checkpoints[5]}": '#8B4513'
    }
    
    # Plot stacked bars
    bottoms = np.zeros(len(checkpoints))
    category_positions = {}  # (checkpoint, category) -> (bottom, height)
    
    for category in categories:
        heights = []
        for checkpoint in checkpoints:
            count = category_counts[checkpoint].get(category, 0)
            heights.append(count)
        
        if sum(heights) > 0:  # Only plot if category has samples
            color = colors.get(category, '#808080')
            bars = ax.bar(x_positions, heights, bar_width, bottom=bottoms, 
                         color=color, label=category.replace('_', ' ').title(),
                         alpha=0.85, edgecolor='white', linewidth=0.5)
            
            # Store positions for flow lines
            for i, checkpoint in enumerate(checkpoints):
                if heights[i] > 0:
                    category_positions[(checkpoint, category)] = (bottoms[i], heights[i])
            
            bottoms += heights
    
    # Customize plot
    ax.set_xlabel('Training Checkpoint', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('OLMo-1B: Evolution of Repetition Behavior Across Training\n' + 
                 'Layer 12 Analysis (75% Depth)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Format checkpoint labels
    checkpoint_labels = [
        cp.replace('step', '').replace('-tokens', '\n').replace('B', 'B tokens')
        for cp in checkpoints
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(checkpoint_labels, fontsize=10)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    create_olmo_alluvial_plot()
