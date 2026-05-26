#!/usr/bin/env python3
"""
Aggregate and plot multihead original vs acquired results across all layers.
Shows which layers/heads have the biggest differences between innate and acquired.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import seaborn as sns


def load_all_layer_data(output_dir, model_name):
    """Load CSV data from all layers."""
    model_short = model_name.split('/')[-1]
    
    all_data = []
    for csv_path in Path(output_dir).glob(f"original_vs_acquired_evolution_{model_short}_L*.csv"):
        layer = int(csv_path.stem.split('_L')[-1])
        df = pd.read_csv(csv_path)
        df['layer'] = layer
        all_data.append(df)
    
    if not all_data:
        print(f"No data found for {model_name}")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(all_data)} layers for {model_short}")
    return combined


def plot_layer_head_heatmap(df, model_name, output_path):
    """
    Create heatmap showing difference (acquired - original) for each layer/head
    at the latest checkpoint.
    """
    model_short = model_name.split('/')[-1]
    
    # Filter to steplatest
    latest = df[df['checkpoint'] == 'steplatest'].copy()
    
    if latest.empty:
        print("No steplatest data found")
        return
    
    # Pivot to get original and acquired contrasts
    orig = latest[latest['category'] == 'original'].copy()
    acq = latest[latest['category'] == 'acquired'].copy()
    
    # Extract layer and head from the 'head' column (format: "L{layer}_H{head}")
    orig['layer_num'] = orig['head'].str.extract(r'L(\d+)')[0].astype(int)
    orig['head_num'] = orig['head'].str.extract(r'H(\d+)')[0].astype(int)
    acq['layer_num'] = acq['head'].str.extract(r'L(\d+)')[0].astype(int)
    acq['head_num'] = acq['head'].str.extract(r'H(\d+)')[0].astype(int)
    
    # Pivot tables
    orig_pivot = orig.pivot_table(index='layer_num', columns='head_num', values='contrast', aggfunc='mean')
    acq_pivot = acq.pivot_table(index='layer_num', columns='head_num', values='contrast', aggfunc='mean')
    
    # Compute difference
    diff_pivot = acq_pivot - orig_pivot
    
    # Determine model params
    n_layers = orig_pivot.index.max() + 1
    n_heads = orig_pivot.columns.max() + 1
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.8], hspace=0.3, wspace=0.3)
    
    # Common colormap settings
    vmax = max(abs(diff_pivot.values.min()), abs(diff_pivot.values.max()))
    
    # Panel 1: Original contrasts heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(orig_pivot, ax=ax1, cmap='Blues', annot=True, fmt='.2f', 
                cbar_kws={'label': 'Contrast'})
    ax1.set_title('Original (Innate) Contrasts', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Head')
    ax1.set_ylabel('Layer')
    
    # Panel 2: Acquired contrasts heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(acq_pivot, ax=ax2, cmap='Reds', annot=True, fmt='.2f',
                cbar_kws={'label': 'Contrast'})
    ax2.set_title('Acquired Contrasts', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Head')
    ax2.set_ylabel('Layer')
    
    # Panel 3: Difference heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(diff_pivot, ax=ax3, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                vmin=-vmax, vmax=vmax, cbar_kws={'label': 'Diff (Acq - Orig)'})
    ax3.set_title('Difference (Acquired - Original)\nGreen=Acquired stronger', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Head')
    ax3.set_ylabel('Layer')
    
    # Panel 4: Bar chart of top differences
    ax4 = fig.add_subplot(gs[1, :])
    
    # Flatten difference matrix and sort
    diff_flat = diff_pivot.stack().reset_index()
    diff_flat.columns = ['layer', 'head', 'diff']
    diff_flat['layer_head'] = diff_flat.apply(lambda r: f"L{int(r['layer'])}_H{int(r['head'])}", axis=1)
    diff_flat_sorted = diff_flat.reindex(diff_flat['diff'].abs().sort_values(ascending=False).index)
    
    # Plot top 16 (or all if fewer)
    top_n = min(16, len(diff_flat_sorted))
    top_diffs = diff_flat_sorted.head(top_n)
    
    colors = ['#27AE60' if d > 0 else '#E74C3C' for d in top_diffs['diff']]
    bars = ax4.barh(range(top_n), top_diffs['diff'].values, color=colors)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(top_diffs['layer_head'].values)
    ax4.invert_yaxis()
    ax4.axvline(x=0, color='gray', linestyle='--')
    ax4.set_xlabel('Difference (Acquired - Original)')
    ax4.set_title('Top Layer-Head Differences (at steplatest)\n'
                  'Green = stronger for acquired, Red = stronger for original (innate)',
                  fontsize=12, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_diffs['diff'].values)):
        ax4.text(val + 0.002 if val > 0 else val - 0.002, i, f'{val:.3f}',
                va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    fig.suptitle(f'{model_short}: Original vs Acquired Attention Head Analysis\n'
                 f'(All Layers at steplatest checkpoint)', fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return diff_flat_sorted


def plot_evolution_grid(df, model_name, output_path):
    """
    Create grid showing evolution across checkpoints for each layer.
    Each row is a layer, showing original vs acquired evolution.
    """
    model_short = model_name.split('/')[-1]
    
    # Get unique layers
    df['layer_num'] = df['head'].str.extract(r'L(\d+)')[0].astype(int)
    layers = sorted(df['layer_num'].unique())
    n_layers = len(layers)
    
    # Custom checkpoint sorting
    def sort_checkpoint(cp):
        if cp == 'steplatest':
            return float('inf')
        return int(cp.replace('step', ''))
    
    checkpoints = sorted(df['checkpoint'].unique(), key=sort_checkpoint)
    checkpoint_labels = []
    for cp in checkpoints:
        label = cp.replace('step', '')
        if label.isdigit():
            num = int(label)
            if num >= 1000:
                label = f"{num // 1000}K"
        checkpoint_labels.append(label)
    
    # Create figure
    fig, axes = plt.subplots(n_layers, 2, figsize=(14, 3 * n_layers), 
                             sharex=True, sharey='row')
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for row, layer in enumerate(layers):
        layer_df = df[df['layer_num'] == layer].copy()
        
        for col, category in enumerate(['original', 'acquired']):
            ax = axes[row, col]
            cat_df = layer_df[layer_df['category'] == category]
            
            if cat_df.empty:
                ax.set_title(f'Layer {layer} - {category.title()}\n(no data)')
                continue
            
            # Map checkpoints to x values
            cp_order = {cp: i for i, cp in enumerate(checkpoints)}
            cat_df = cat_df.copy()
            cat_df['cp_idx'] = cat_df['checkpoint'].map(cp_order)
            
            # Plot each head
            heads = cat_df['head'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(heads))))
            
            for i, head in enumerate(sorted(heads)):
                head_df = cat_df[cat_df['head'] == head]
                head_num = int(head.split('_H')[1])
                ax.plot(head_df['cp_idx'], head_df['contrast'], 
                       marker='o', markersize=4, linewidth=1.5,
                       color=colors[head_num % len(colors)], 
                       label=f'H{head_num}', alpha=0.8)
            
            ax.set_title(f'Layer {layer} - {category.title()}', fontsize=11, fontweight='bold',
                        color='#4A90E2' if category == 'original' else '#E74C3C')
            ax.set_xticks(range(len(checkpoints)))
            ax.set_xticklabels(checkpoint_labels)
            ax.grid(True, alpha=0.3)
            
            if col == 0:
                ax.set_ylabel('Contrast')
            if row == n_layers - 1:
                ax.set_xlabel('Checkpoint')
            
            # Only show legend for first row
            if row == 0:
                ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    fig.suptitle(f'{model_short}: Attention Head Contrast Evolution by Layer\n'
                 f'Original (Innate) vs Acquired', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='EleutherAI/pythia-70m')
    parser.add_argument('--output-dir', default='/home/mmahaut/projects/parrots/outputs_multihead_full')
    args = parser.parse_args()
    
    # Load data
    df = load_all_layer_data(args.output_dir, args.model_name)
    if df is None:
        return
    
    model_short = args.model_name.split('/')[-1]
    
    # Plot 1: Heatmap summary
    heatmap_path = f"{args.output_dir}/original_vs_acquired_all_layers_{model_short}_heatmap.png"
    diff_sorted = plot_layer_head_heatmap(df, args.model_name, heatmap_path)
    
    # Plot 2: Evolution grid
    grid_path = f"{args.output_dir}/original_vs_acquired_all_layers_{model_short}_evolution.png"
    plot_evolution_grid(df, args.model_name, grid_path)
    
    # Print top findings
    if diff_sorted is not None:
        print("\nTop differences (Acquired - Original):")
        print("  Positive = stronger for acquired repetition")
        print("  Negative = stronger for original (innate) repetition")
        print("-" * 50)
        for _, row in diff_sorted.head(10).iterrows():
            direction = "acquired" if row['diff'] > 0 else "original" 
            print(f"  {row['layer_head']}: {row['diff']:+.4f} (stronger for {direction})")


if __name__ == "__main__":
    main()
