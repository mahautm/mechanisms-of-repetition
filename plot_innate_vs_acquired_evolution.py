#!/usr/bin/env python3
"""
Plot original vs acquired head contrast evolution in the same style as 
multihead_cycle_evolution.png:

- Horizontal subplots for each checkpoint
- X-axis: Checkpoint (training step)
- Y-axis: Contrast (expected - alternative logit)
- Lines: Different attention heads (layer_head)
- Alpha: 0.1 for all heads (background), full opacity for top interesting heads
- Two sections: "Innate (Original)" and "Acquired"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import argparse


def load_all_layer_data(output_dir, model_name):
    """Load CSV data from all layers."""
    model_short = model_name.split('/')[-1]
    
    all_data = []
    for csv_path in Path(output_dir).glob(f"original_vs_acquired_evolution_{model_short}_L*.csv"):
        df = pd.read_csv(csv_path)
        all_data.append(df)
    
    if not all_data:
        print(f"No data found for {model_name}")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(all_data)} layer files for {model_short}")
    return combined


def plot_innate_vs_acquired_evolution(df, model_name, output_path, max_heads_to_show=8, background_alpha=0.08):
    """
    Create horizontal plots in the style of multihead_cycle_evolution.png
    
    Layout:
    - Left section: "Innate" - subplots for each checkpoint showing ORIGINAL samples
    - Right section: "Acquired" - subplots for each checkpoint showing ACQUIRED samples
    
    Each subplot:
    - X-axis: Checkpoint (training step)
    - Y-axis: Contrast value
    - Lines: Different heads, alpha=0.1 for background, full for interesting
    """
    
    model_short = model_name.split('/')[-1]
    
    # Checkpoint ordering
    def sort_checkpoint(cp):
        if cp == 'steplatest':
            return float('inf')
        return int(cp.replace('step', ''))
    
    checkpoints = sorted(df['checkpoint'].unique(), key=sort_checkpoint)
    
    # Checkpoint labels (1, 1K, 10K, 100K, latest)
    checkpoint_labels = {}
    for cp in checkpoints:
        label = cp.replace('step', '')
        if label.isdigit():
            num = int(label)
            if num >= 1000:
                label = f"{num // 1000}K"
        elif label == 'latest':
            label = 'latest'
        checkpoint_labels[cp] = label
    
    # Map checkpoints to x positions
    checkpoint_to_x = {cp: i for i, cp in enumerate(checkpoints)}
    df['x'] = df['checkpoint'].map(checkpoint_to_x)
    
    # Split by category
    original_df = df[df['category'] == 'original'].copy()
    acquired_df = df[df['category'] == 'acquired'].copy()
    
    if original_df.empty and acquired_df.empty:
        print("No data to plot")
        return
    
    # Find globally interesting heads (highest variance across both categories)
    all_head_variances = df.groupby('head')['contrast'].var().sort_values(ascending=False)
    global_interesting_heads = all_head_variances.head(max_heads_to_show).index.tolist()
    
    # Color and style mapping
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    head_color_map = {}
    head_style_map = {}
    for i, head in enumerate(global_interesting_heads):
        head_color_map[head] = colors[i % len(colors)]
        head_style_map[head] = styles[i % len(styles)]
    
    # Calculate shared y-axis limits
    y_min = df['contrast'].min()
    y_max = df['contrast'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1
    y_min_padded = y_min - y_padding
    y_max_padded = y_max + y_padding
    
    # Create figure with 2 panels: Innate and Acquired
    # Similar to ICL vs Natural layout
    fig_width = 12
    fig = plt.figure(figsize=(fig_width, 4))
    
    # Use gridspec: [Innate panel] [small gap] [Acquired panel]
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.1)
    
    ax_innate = fig.add_subplot(gs[0, 0])
    ax_acquired = fig.add_subplot(gs[0, 2])
    
    # Style the panels
    ax_innate.set_facecolor('#e8f4e8')  # Light green for innate
    ax_acquired.set_facecolor('#e8e8f4')  # Light blue for acquired
    
    for ax, cat_df, title, border_color in [
        (ax_innate, original_df, 'Innate (Original)', '#27AE60'),
        (ax_acquired, acquired_df, 'Acquired', '#3498DB')
    ]:
        if cat_df.empty:
            ax.set_title(f'{title}\n(no data)', fontsize=14, fontweight='bold')
            ax.set_xlim(-0.5, len(checkpoints) - 0.5)
            ax.set_ylim(y_min_padded, y_max_padded)
            continue
        
        # Plot ALL heads in light grey - background
        for head in cat_df['head'].unique():
            head_data = cat_df[cat_df['head'] == head]
            # Sort by x for proper line drawing
            head_data = head_data.sort_values('x')
            ax.plot(head_data['x'], head_data['contrast'],
                   color='lightgrey', alpha=background_alpha, linewidth=1)
        
        # Highlight interesting heads with full opacity and colors
        for head in global_interesting_heads:
            head_data = cat_df[cat_df['head'] == head]
            if not head_data.empty:
                head_data = head_data.sort_values('x')
                ax.plot(head_data['x'], head_data['contrast'],
                       color=head_color_map[head], linestyle=head_style_map[head],
                       marker='o', markersize=6, linewidth=2.5, label=head)
        
        # Styling
        ax.set_title(title, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=border_color, alpha=0.3))
        ax.set_xlabel("Training Checkpoint", fontsize=12)
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels([checkpoint_labels[cp] for cp in checkpoints], fontsize=10)
        ax.set_ylim(y_min_padded, y_max_padded)
        ax.grid(True, alpha=0.3)
        
        # Border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(border_color)
    
    # Y-axis label only on left
    ax_innate.set_ylabel("Contrast (expected - alternative)", fontsize=12)
    ax_acquired.set_ylabel("")
    
    # Create legend below
    if global_interesting_heads:
        legend_elements = []
        for head in global_interesting_heads:
            legend_elements.append(plt.Line2D([0], [0], color=head_color_map[head],
                                            linestyle=head_style_map[head], marker='o',
                                            markersize=6, linewidth=2, label=head))
        
        ncol = min(len(legend_elements), 4)
        fig.legend(handles=legend_elements, loc='lower center', ncol=ncol,
                  bbox_to_anchor=(0.5, -0.15), fontsize=10)
    
    fig.suptitle(f'{model_short}: Attention Head Contrast Evolution\n'
                 f'Innate (Original) vs Acquired Repetition',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, top=0.85)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='EleutherAI/pythia-70m')
    parser.add_argument('--output-dir', default='/home/mmahaut/projects/parrots/outputs_multihead_full')
    parser.add_argument('--max-heads', type=int, default=8)
    parser.add_argument('--background-alpha', type=float, default=0.08)
    args = parser.parse_args()
    
    # Load data
    df = load_all_layer_data(args.output_dir, args.model_name)
    if df is None:
        return
    
    model_short = args.model_name.split('/')[-1]
    output_path = f"{args.output_dir}/innate_vs_acquired_evolution_{model_short}.png"
    
    plot_innate_vs_acquired_evolution(
        df,
        args.model_name,
        output_path,
        args.max_heads,
        args.background_alpha
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
