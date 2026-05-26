#!/usr/bin/env python3
"""
Plot original vs acquired repetition evolution across training checkpoints
Similar to multihead_cycle_evolution.png but tracking:
- Original: sequences that were already repeating at step1
- Acquired: sequences that started repeating at later checkpoints
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

def load_cycle_evolution_data(results_file):
    """Load cycle evolution JSON data"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return {cp: {int(k): v for k, v in status.items()} 
            for cp, status in data.items()}

def categorize_sequences(all_status, checkpoints):
    """
    Categorize each sequence:
    - original: repeating at step1
    - acquired: first repeating at a later checkpoint
    - never: never repeats
    """
    n_texts = max(max(status.keys()) for status in all_status.values()) + 1
    
    categories = {}
    first_repetition = {}
    
    for idx in range(n_texts):
        first_rep_cp = None
        for cp in checkpoints:
            if cp in all_status and idx in all_status[cp]:
                if all_status[cp][idx]:  # is repeating
                    first_rep_cp = cp
                    break
        
        first_repetition[idx] = first_rep_cp
        
        if first_rep_cp is None:
            categories[idx] = 'never'
        elif first_rep_cp == checkpoints[0]:  # step1
            categories[idx] = 'original'
        else:
            categories[idx] = 'acquired'
    
    return categories, first_repetition

def compute_counts_at_checkpoint(all_status, checkpoints, categories, checkpoint):
    """Compute how many original vs acquired sequences are repeating at a given checkpoint"""
    counts = {
        'original_repeating': 0,
        'original_not_repeating': 0,
        'acquired_repeating': 0,
        'acquired_not_repeating': 0,
        'never': 0
    }
    
    if checkpoint not in all_status:
        return counts
    
    for idx, cat in categories.items():
        if idx not in all_status[checkpoint]:
            continue
            
        is_repeating = all_status[checkpoint][idx]
        
        if cat == 'never':
            counts['never'] += 1
        elif cat == 'original':
            if is_repeating:
                counts['original_repeating'] += 1
            else:
                counts['original_not_repeating'] += 1
        else:  # acquired
            if is_repeating:
                counts['acquired_repeating'] += 1
            else:
                counts['acquired_not_repeating'] += 1
    
    return counts

def create_evolution_plot(models_data, output_path, show_title=False):
    """
    Create side-by-side evolution plot for multiple models
    Similar style to multihead_cycle_evolution.png
    """
    n_models = len(models_data)
    
    # Paper-ready styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
    
    # Colors
    colors = {
        'original_repeating': '#4A90E2',      # Blue - original & repeating
        'original_not_repeating': '#A8D0FF',  # Light blue - original but stopped
        'acquired_repeating': '#E67E22',      # Orange - acquired & repeating
        'acquired_not_repeating': '#FFD4A8',  # Light orange - acquired but stopped
        'never': '#D0D0D0'                    # Grey - never repeats
    }
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    
    for ax_idx, (model_name, model_info) in enumerate(models_data.items()):
        ax = axes[ax_idx]
        
        all_status = model_info['status']
        checkpoints = model_info['checkpoints']
        
        # Get categories
        categories, first_repetition = categorize_sequences(all_status, checkpoints)
        
        n_texts = len(categories)
        n_original = sum(1 for c in categories.values() if c == 'original')
        n_acquired = sum(1 for c in categories.values() if c == 'acquired')
        n_never = sum(1 for c in categories.values() if c == 'never')
        
        print(f"\n{model_name}:")
        print(f"  Total: {n_texts}")
        print(f"  Original (step1): {n_original} ({100*n_original/n_texts:.1f}%)")
        print(f"  Acquired (later): {n_acquired} ({100*n_acquired/n_texts:.1f}%)")
        print(f"  Never repeats: {n_never} ({100*n_never/n_texts:.1f}%)")
        
        # Compute counts at each checkpoint
        checkpoint_labels = []
        original_rep = []
        original_not = []
        acquired_rep = []
        acquired_not = []
        never_counts = []
        
        for cp in checkpoints:
            counts = compute_counts_at_checkpoint(all_status, checkpoints, categories, cp)
            
            # Label formatting
            if cp == 'steplatest':
                label = 'Latest'
            else:
                num = cp.replace('step', '')
                if len(num) >= 4:
                    label = f'{int(num)//1000}K'
                else:
                    label = num
            checkpoint_labels.append(label)
            
            original_rep.append(counts['original_repeating'])
            original_not.append(counts['original_not_repeating'])
            acquired_rep.append(counts['acquired_repeating'])
            acquired_not.append(counts['acquired_not_repeating'])
            never_counts.append(counts['never'])
        
        # Convert to proportions
        original_rep = np.array(original_rep) / n_texts
        original_not = np.array(original_not) / n_texts
        acquired_rep = np.array(acquired_rep) / n_texts
        acquired_not = np.array(acquired_not) / n_texts
        never_counts = np.array(never_counts) / n_texts
        
        x = np.arange(len(checkpoints))
        width = 0.7
        
        # Stacked bar chart
        # Bottom: never (grey), then acquired not repeating, acquired repeating, original not repeating, original repeating
        bottom = np.zeros(len(checkpoints))
        
        # Never repeats (grey)
        ax.bar(x, never_counts, width, bottom=bottom, color=colors['never'], 
               label='Never repeats', edgecolor='white', linewidth=0.5)
        bottom += never_counts
        
        # Acquired - not repeating (light orange)
        ax.bar(x, acquired_not, width, bottom=bottom, color=colors['acquired_not_repeating'],
               label='Acquired (not rep.)', edgecolor='white', linewidth=0.5)
        bottom += acquired_not
        
        # Acquired - repeating (orange)
        ax.bar(x, acquired_rep, width, bottom=bottom, color=colors['acquired_repeating'],
               label='Acquired (repeating)', edgecolor='white', linewidth=0.5)
        bottom += acquired_rep
        
        # Original - not repeating (light blue)
        ax.bar(x, original_not, width, bottom=bottom, color=colors['original_not_repeating'],
               label='Original (not rep.)', edgecolor='white', linewidth=0.5)
        bottom += original_not
        
        # Original - repeating (blue)
        ax.bar(x, original_rep, width, bottom=bottom, color=colors['original_repeating'],
               label='Original (repeating)', edgecolor='white', linewidth=0.5)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoint_labels, rotation=45, ha='right')
        ax.set_xlabel('Training Step', fontweight='bold')
        
        if ax_idx == 0:
            ax.set_ylabel('Proportion of Sequences', fontweight='bold')
        
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.5, len(checkpoints) - 0.5)
        
        # Model name as title
        short_name = model_name.split('/')[-1]
        ax.set_title(short_name, fontweight='bold', fontsize=13)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # Reverse order for legend (top to bottom matches visual)
    fig.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=5, fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.92)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Plot saved to {output_path}")
    
    # PDF version
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"📊 PDF saved to {pdf_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./outputs_multihead_full")
    parser.add_argument("--results-dir", type=str, default="./cycle_evolution_results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data for both models
    models_data = {}
    
    checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
    
    # Pythia-1.4b
    file_1_4b = results_dir / "cycle_evolution_status_EleutherAI_pythia-1.4b.json"
    if file_1_4b.exists():
        models_data['EleutherAI/pythia-1.4b'] = {
            'status': load_cycle_evolution_data(file_1_4b),
            'checkpoints': checkpoints
        }
        print(f"Loaded Pythia-1.4b data")
    
    # Pythia-70m
    file_70m = results_dir / "cycle_evolution_status_EleutherAI_pythia-70m.json"
    if file_70m.exists():
        models_data['EleutherAI/pythia-70m'] = {
            'status': load_cycle_evolution_data(file_70m),
            'checkpoints': checkpoints
        }
        print(f"Loaded Pythia-70m data")
    
    if not models_data:
        print("No data files found!")
        return
    
    # Create plot
    output_path = output_dir / "original_vs_acquired_repetition_evolution.png"
    create_evolution_plot(models_data, output_path)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
