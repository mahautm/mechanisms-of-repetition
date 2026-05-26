#!/usr/bin/env python3
"""
Combine cycle iteration results from individual checkpoint runs and create stacked grid plot.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse


def plot_cycle_iteration_grid(results_df, model_name, output_path, checkpoints, max_heads_to_show=8):
    """
    Plot cycle-iteration evolution with checkpoints stacked in rows.
    - Rows: checkpoints
    - Columns: Innate (Original) vs Acquired
    - X-axis: cycle iteration (0..N)
    - Y-axis: contrast
    """
    if results_df.empty:
        print("No data to plot")
        return

    iterations = sorted(results_df['iteration'].unique())

    head_variances = results_df.groupby('head')['contrast'].var().sort_values(ascending=False)
    interesting_heads = head_variances.head(max_heads_to_show).index.tolist()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    head_color_map = {head: colors[i % len(colors)] for i, head in enumerate(interesting_heads)}
    head_style_map = {head: styles[i % len(styles)] for i, head in enumerate(interesting_heads)}

    y_min = results_df['contrast'].min()
    y_max = results_df['contrast'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1 if y_range > 0 else 0.1
    y_min_padded = y_min - y_padding
    y_max_padded = y_max + y_padding

    n_rows = len(checkpoints)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows), sharex=True, sharey=True)

    if n_rows == 1:
        axes = np.array([axes])

    def checkpoint_label(cp):
        label = cp.replace('step', '')
        if label.isdigit():
            num = int(label)
            if num >= 1000:
                label = f"{num // 1000}K"
        elif label == 'latest':
            label = 'latest'
        return label

    background_alpha = 0.08

    for row_idx, checkpoint in enumerate(checkpoints):
        row_label = checkpoint_label(checkpoint)
        for col_idx, (category, title, border_color, bg_color) in enumerate([
            ('original', 'Innate (Original)', '#4A90E2', '#e8f4fc'),
            ('acquired', 'Acquired', '#E74C3C', '#fce8e8')
        ]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(bg_color)

            cat_df = results_df[(results_df['checkpoint'] == checkpoint) & (results_df['category'] == category)]

            if cat_df.empty:
                ax.set_title(f'{row_label} - {title}\n(no data)', fontsize=12, fontweight='bold')
                continue

            for head in cat_df['head'].unique():
                head_data = cat_df[cat_df['head'] == head].sort_values('iteration')
                ax.plot(head_data['iteration'], head_data['contrast'],
                        color='lightgrey', alpha=background_alpha, linewidth=1)

            interesting_df = cat_df[cat_df['head'].isin(interesting_heads)]
            if not interesting_df.empty:
                for head in interesting_heads:
                    head_data = interesting_df[interesting_df['head'] == head].sort_values('iteration')
                    if not head_data.empty:
                        ax.plot(head_data['iteration'], head_data['contrast'],
                                color=head_color_map[head], linestyle=head_style_map[head],
                                marker='o', markersize=6, linewidth=2, label=head)

            ax.set_title(f'{row_label} - {title}', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=border_color, alpha=0.2))
            ax.set_xticks(iterations)
            ax.set_xticklabels([str(i) for i in iterations], fontsize=9)
            ax.set_ylim(y_min_padded, y_max_padded)
            ax.grid(True, alpha=0.3)

            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color(border_color)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Cycle Iteration", fontsize=10)

            if col_idx == 0:
                ax.set_ylabel("Contrast (expected - alternative)", fontsize=10)

    if interesting_heads:
        legend_elements = [
            plt.Line2D([0], [0], color=head_color_map[head],
                       linestyle=head_style_map[head], marker='o',
                       markersize=5, linewidth=2, label=head)
            for head in interesting_heads
        ]
        ncol = min(len(legend_elements), 4)
        fig.legend(handles=legend_elements, loc='lower center', ncol=ncol,
                   bbox_to_anchor=(0.5, -0.02), fontsize=10)

    model_short = model_name.split('/')[-1]
    fig.suptitle(f'{model_short} All layers: Cycle Iteration Evolution\n'
                 f'Checkpoints stacked by row (Innate vs Acquired)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.90)

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='EleutherAI/pythia-70m')
    parser.add_argument('--output-dir', default='/home/mmahaut/projects/parrots/outputs_multihead_full')
    parser.add_argument('--checkpoints', default='step1,step1000,step5000,step10000,step100000,steplatest')
    args = parser.parse_args()
    
    model_short = args.model_name.split('/')[-1]
    output_dir = Path(args.output_dir)
    
    checkpoints = [cp.strip() for cp in args.checkpoints.split(',')]
    
    # Sort checkpoints
    def sort_checkpoint_key(cp):
        if cp == 'steplatest':
            return float('inf')
        if cp.startswith('step'):
            try:
                return int(cp[4:])
            except ValueError:
                return 0
        return 0
    
    checkpoints = sorted(checkpoints, key=sort_checkpoint_key)
    
    # Load individual checkpoint CSVs (with optional chunk suffix)
    all_dfs = []
    for checkpoint in checkpoints:
        chunk_paths = list(output_dir.glob(f"cycle_iteration_evolution_{model_short}_all_layers_{checkpoint}_chunk*.csv"))
        if chunk_paths:
            for csv_path in chunk_paths:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
                print(f"Loaded: {csv_path}")
            continue

        csv_path = output_dir / f"cycle_iteration_evolution_{model_short}_all_layers_{checkpoint}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            print(f"Loaded: {csv_path}")
        else:
            print(f"Missing: {csv_path}")
    
    if not all_dfs:
        print("No data files found!")
        return
    
    # Combine (weighted by n_samples across chunks if present)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if 'n_samples' in combined_df.columns:
        combined_df = (
            combined_df
            .assign(weight=combined_df['n_samples'])
            .groupby(['checkpoint', 'iteration', 'head', 'category'], as_index=False)
            .apply(lambda g: pd.Series({
                'contrast': (g['contrast'] * g['weight']).sum() / g['weight'].sum(),
                'n_samples': g['weight'].sum()
            }))
        )
    
    # Save combined CSV
    combined_csv = output_dir / f"cycle_iteration_grid_{model_short}_all_layers.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nSaved combined: {combined_csv}")
    
    # Generate plot
    output_path = output_dir / f"cycle_iteration_grid_{model_short}_all_layers.png"
    plot_cycle_iteration_grid(combined_df, args.model_name, str(output_path), checkpoints)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
