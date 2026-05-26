#!/usr/bin/env python3
"""
Script to replot entropy evolution excluding ICL data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_entropy_evolution_no_icl(df, save_path=None):
    """
    Create paper-ready boxplot showing entropy distributions across cycles (excluding ICL data)
    """
    # Filter out ICL data
    df_filtered = df[df['data_type'] != 'icl'].copy()
    
    # Set up the plot style for publication
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # Single plot - boxplots with paired layout (original seaborn style)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    # Use seaborn boxplot with hue for paired layout (two boxes per cycle)
    # Color scheme: Natural (red) and ICL (blue) to match other plots
    sns.boxplot(data=df_filtered, x='cycle', y='entropy_values', hue='data_type', ax=ax,
                palette={'natural': '#e74c3c', 'no_cycle_icl': '#3498db'})
    
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Entropy (nats)', fontsize=12)
    
    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Natural', 'No-Cycle ICL'], fontsize=11, frameon=True, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig

if __name__ == "__main__":
    # Load the existing data
    df = pd.read_csv("/home/mmahaut/projects/parrots/logit_entropy_results.csv")
    
    print("Data types available:", df['data_type'].unique())
    print("Cycles available:", sorted(df['cycle'].unique()))
    
    # Print summary statistics excluding ICL
    df_no_icl = df[df['data_type'] != 'icl']
    print("\nSummary Statistics (excluding ICL):")
    summary = df_no_icl.groupby(['cycle', 'data_type'])['entropy_values'].agg(['count', 'mean', 'std']).round(4)
    print(summary)
    
    # Create the plot
    fig = plot_entropy_evolution_no_icl(df, save_path="/home/mmahaut/projects/parrots/logit_entropy_evolution_no_icl.png")