#!/usr/bin/env python3
"""
Analyze multihead attention differences between original (innate) vs acquired repetition.
Uses existing cycle evolution data to classify samples, then computes head-level statistics.

This version doesn't re-run inference - it uses the existing multihead analysis outputs
and computes the difference based on which samples are classified as original vs acquired.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse

def load_cycle_evolution_data(results_file):
    """Load cycle evolution JSON data"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return {cp: {int(k): v for k, v in status.items()} 
            for cp, status in data.items()}

def classify_samples(all_status, checkpoints):
    """
    Classify each sample as original, acquired, or never-repeating.
    
    Returns dict: {sample_idx: {'category': 'original'|'acquired'|'never', 'first_cp': checkpoint}}
    """
    n_texts = max(max(status.keys()) for status in all_status.values() if status) + 1
    
    classification = {}
    
    for idx in range(n_texts):
        first_rep_cp = None
        for cp in checkpoints:
            if cp in all_status and idx in all_status[cp]:
                if all_status[cp][idx]:  # is repeating
                    first_rep_cp = cp
                    break
        
        if first_rep_cp is None:
            classification[idx] = {'category': 'never', 'first_cp': None}
        elif first_rep_cp == checkpoints[0]:  # step1
            classification[idx] = {'category': 'original', 'first_cp': first_rep_cp}
        else:
            classification[idx] = {'category': 'acquired', 'first_cp': first_rep_cp}
    
    return classification

def parse_multihead_output(output_file, layer):
    """Parse multihead analysis output to get repetition index."""
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Extract heatmap (aggregated across all samples)
    heatmap_match = re.search(rf'layer {layer} natural heatmap: \[(.*?)\]', content, re.DOTALL)
    if heatmap_match:
        heatmap_str = heatmap_match.group(1).replace('\n', ' ')
        heatmap = np.array([float(x.strip()) for x in heatmap_str.split(',') if x.strip()])
    else:
        heatmap = None
    
    # Extract data index
    data_match = re.search(rf'layer {layer} data index: \[(.*?)\]', content, re.DOTALL)
    if data_match:
        data_str = data_match.group(1).replace('\n', ' ')
        data_index = [int(x.strip()) for x in data_str.split(',') if x.strip()]
    else:
        data_index = None
    
    # Extract repetition index (where cycles start - 500 means no cycle)
    rep_match = re.search(rf'layer {layer} repetition index: \[(.*?)\]', content, re.DOTALL)
    if rep_match:
        rep_str = rep_match.group(1).replace('\n', ' ')
        rep_index = [int(x.strip()) for x in rep_str.split(',') if x.strip()]
    else:
        rep_index = None
    
    return heatmap, data_index, rep_index

def compute_category_stats(classification, data_index, rep_index):
    """
    Compute statistics by category (original vs acquired).
    
    Note: 500 in rep_index means no cycle detected in that sample.
    """
    original_repeating = []
    original_not_repeating = []
    acquired_repeating = []
    acquired_not_repeating = []
    
    for i, idx in enumerate(data_index):
        if idx not in classification:
            continue
        
        cat = classification[idx]['category']
        is_repeating = rep_index[i] != 500  # 500 means no cycle
        
        if cat == 'original':
            if is_repeating:
                original_repeating.append(idx)
            else:
                original_not_repeating.append(idx)
        elif cat == 'acquired':
            if is_repeating:
                acquired_repeating.append(idx)
            else:
                acquired_not_repeating.append(idx)
    
    return {
        'original_repeating': original_repeating,
        'original_not_repeating': original_not_repeating,
        'acquired_repeating': acquired_repeating,
        'acquired_not_repeating': acquired_not_repeating
    }

def plot_category_evolution(models_data, output_path):
    """
    Plot evolution of original vs acquired across checkpoints.
    Shows how each category behaves across training.
    """
    n_models = len(models_data)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
    })
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax_idx, (model_name, model_info) in enumerate(models_data.items()):
        ax = axes[ax_idx]
        checkpoints = model_info['checkpoints']
        stats_by_cp = model_info['stats']
        
        # Compute percentages
        x_labels = []
        orig_rep_pct = []
        acq_rep_pct = []
        
        for cp in checkpoints:
            if cp not in stats_by_cp:
                continue
            
            stats = stats_by_cp[cp]
            total_orig = len(stats['original_repeating']) + len(stats['original_not_repeating'])
            total_acq = len(stats['acquired_repeating']) + len(stats['acquired_not_repeating'])
            
            if total_orig > 0:
                orig_rep_pct.append(100 * len(stats['original_repeating']) / total_orig)
            else:
                orig_rep_pct.append(0)
            
            if total_acq > 0:
                acq_rep_pct.append(100 * len(stats['acquired_repeating']) / total_acq)
            else:
                acq_rep_pct.append(0)
            
            # Format checkpoint label
            if cp.startswith('step'):
                step_num = cp.replace('step', '')
                if step_num == 'latest':
                    x_labels.append('Latest')
                elif int(step_num) >= 1000:
                    x_labels.append(f'{int(step_num)//1000}K')
                else:
                    x_labels.append(step_num)
            else:
                x_labels.append(cp)
        
        x = np.arange(len(x_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_rep_pct, width, label='Original (innate)', color='#4A90E2', alpha=0.8)
        bars2 = ax.bar(x + width/2, acq_rep_pct, width, label='Acquired', color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Training Checkpoint')
        ax.set_ylabel('% Still Repeating')
        ax.set_title(model_name.split('/')[-1])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Original vs Acquired Repetition Behavior Across Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/home/mmahaut/projects/parrots/outputs_multihead_full')
    args = parser.parse_args()
    
    checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
    
    models = [
        {'name': 'EleutherAI/pythia-70m', 'layer': 4},
        {'name': 'EleutherAI/pythia-1.4b', 'layer': 19},
    ]
    
    models_data = {}
    
    for model_info in models:
        model_name = model_info['name']
        layer = model_info['layer']
        model_short = model_name.split('/')[-1]
        clean_model = model_name.replace('/', '_')
        
        # Load cycle evolution data
        cycle_file = Path(f'/home/mmahaut/projects/parrots/cycle_evolution_results/cycle_evolution_status_{clean_model}.json')
        
        if not cycle_file.exists():
            print(f"Cycle evolution file not found: {cycle_file}")
            continue
        
        all_status = load_cycle_evolution_data(cycle_file)
        classification = classify_samples(all_status, checkpoints)
        
        n_original = sum(1 for c in classification.values() if c['category'] == 'original')
        n_acquired = sum(1 for c in classification.values() if c['category'] == 'acquired')
        n_never = sum(1 for c in classification.values() if c['category'] == 'never')
        print(f"\n{model_name}: {n_original} original, {n_acquired} acquired, {n_never} never")
        
        # Load multihead analysis data for each checkpoint
        stats_by_cp = {}
        for cp in checkpoints:
            output_file = Path(args.output_dir) / model_name / cp / f"layer_{layer}" / "full_analysis_cyc0_ml32.out"
            
            if not output_file.exists():
                print(f"  {cp}: output not found")
                continue
            
            heatmap, data_index, rep_index = parse_multihead_output(output_file, layer)
            
            if data_index is None or rep_index is None:
                print(f"  {cp}: could not parse output")
                continue
            
            stats = compute_category_stats(classification, data_index, rep_index)
            stats_by_cp[cp] = stats
            
            n_orig_rep = len(stats['original_repeating'])
            n_orig_not = len(stats['original_not_repeating'])
            n_acq_rep = len(stats['acquired_repeating'])
            n_acq_not = len(stats['acquired_not_repeating'])
            
            print(f"  {cp}: orig_rep={n_orig_rep}, orig_not={n_orig_not}, acq_rep={n_acq_rep}, acq_not={n_acq_not}")
        
        models_data[model_name] = {
            'checkpoints': checkpoints,
            'stats': stats_by_cp,
            'classification': classification
        }
    
    # Generate plot
    if models_data:
        output_path = Path(args.output_dir) / 'original_vs_acquired_behavior_evolution.png'
        plot_category_evolution(models_data, str(output_path))

if __name__ == '__main__':
    main()
