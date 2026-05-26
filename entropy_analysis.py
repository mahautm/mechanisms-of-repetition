#!/usr/bin/env python3
"""
Entropy Analysis for Repetitive Cycles

This script analyzes the entropy of contrast values (as proxy for probability distributions) 
in repetitive cycles for both ICL (in-context learning) and natural data across cycles 0-4.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def parse_heatmap_line(line: str) -> List[float]:
    """
    Parse a heatmap line to extract the numerical values.
    
    Args:
        line: Line containing heatmap data like "layer 0 natural heatmap: [value1, value2, ...]"
        
    Returns:
        List of float values
    """
    try:
        # Find the part after the colon
        parts = line.split(': ')
        if len(parts) < 2:
            return []
            
        # Extract the array part
        array_str = parts[1].strip()
        if not (array_str.startswith('[') and array_str.endswith(']')):
            return []
            
        # Remove brackets and split by comma
        array_str = array_str[1:-1]  # Remove [ and ]
        values = []
        
        for val_str in array_str.split(','):
            val_str = val_str.strip()
            if val_str:
                try:
                    # Handle scientific notation
                    values.append(float(val_str))
                except ValueError:
                    continue
                    
        return values
        
    except Exception:
        return []


def parse_contrast_analysis_output(file_path: str) -> Optional[Dict]:
    """
    Parse the output from contrast_analysis to extract heatmap values.
    
    Args:
        file_path: Path to the .out file containing analysis results
        
    Returns:
        Dictionary containing extracted heatmap data or None if parsing fails
    """
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        data = {}
        
        for line in lines:
            line = line.strip()
            
            # Look for heatmap lines
            if 'heatmap:' in line:
                if 'natural heatmap:' in line:
                    values = parse_heatmap_line(line)
                    if values:
                        data['natural'] = values
                elif 'icl heatmap:' in line and 'no-cycle' not in line:
                    values = parse_heatmap_line(line)
                    if values:
                        data['icl'] = values
                elif 'no-cycle icl heatmap:' in line:
                    values = parse_heatmap_line(line)
                    if values:
                        data['no_cycle_icl'] = values
                        
        return data if data else None
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def compute_entropy_from_values(values: List[float]) -> float:
    """
    Compute entropy from a list of contrast values.
    
    We'll treat the absolute values of contrasts as weights and compute 
    the entropy of their normalized distribution.
    
    Args:
        values: List of contrast values
        
    Returns:
        Entropy value (float)
    """
    if not values:
        return 0.0
        
    # Convert to numpy array and take absolute values
    arr = np.array(values)
    abs_values = np.abs(arr)
    
    # Normalize to get a probability distribution
    total = np.sum(abs_values)
    if total == 0:
        return 0.0
        
    probs = abs_values / total
    
    # Compute entropy: H(p) = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -np.sum(probs * np.log(probs + epsilon))
    
    return entropy



def analyze_entropy_across_cycles(base_path: str, layers: List[int], cycles: List[int]) -> pd.DataFrame:
    """
    Analyze entropy across cycles for specified layers.
    
    Args:
        base_path: Base directory path
        layers: List of layer numbers to analyze
        cycles: List of cycle numbers to analyze
        
    Returns:
        DataFrame with entropy results
    """
    results = []
    
    for layer in layers:
        for cycle in cycles:
            cycle_file = f"{base_path}/layer_{layer}/full_analysis_cyc{cycle}_ml32.out"
            
            if os.path.exists(cycle_file):
                # Parse the actual output file
                parsed_data = parse_contrast_analysis_output(cycle_file)
                
                if parsed_data:
                    # Compute entropy for each data type
                    for data_type, values in parsed_data.items():
                        entropy_value = compute_entropy_from_values(values)
                        
                        results.append({
                            'layer': layer,
                            'cycle': cycle,
                            'data_type': data_type,
                            'entropy': entropy_value,
                            'file_path': cycle_file,
                            'num_heads': len(values)
                        })
                else:
                    print(f"Could not parse data for layer {layer}, cycle {cycle}")
            else:
                print(f"File not found: {cycle_file}")
    
    return pd.DataFrame(results)


def plot_entropy_evolution(df: pd.DataFrame, output_path: str = None):
    """
    Plot entropy evolution across cycles.
    
    Args:
        df: DataFrame with entropy results
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Filter for specific data types we're interested in
    data_types = ['no_cycle_icl', 'natural']
    colors = ['#1f77b4', '#ff7f0e']  # Blue for no_cycle_icl, orange for natural
    
    for i, data_type in enumerate(data_types):
        subset = df[df['data_type'] == data_type]
        
        if not subset.empty:
            # Group by cycle and compute mean entropy across layers
            cycle_means = subset.groupby('cycle')['entropy'].mean().reset_index()
            cycle_stds = subset.groupby('cycle')['entropy'].std().reset_index()
            
            plt.errorbar(cycle_means['cycle'], cycle_means['entropy'], 
                        yerr=cycle_stds['entropy'], 
                        label=f'{data_type.replace("_", " ").title()}',
                        color=colors[i], marker='o', markersize=6, linewidth=2)
    
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('Entropy (bits)', fontsize=12)
    plt.title('Entropy Evolution Across Repetitive Cycles\n(Average across all layers)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def plot_entropy_by_layer(df: pd.DataFrame, output_path: str = None):
    """
    Plot entropy evolution by layer using matplotlib instead of seaborn.
    
    Args:
        df: DataFrame with entropy results
        output_path: Optional path to save the plot
    """
    data_types = ['no_cycle_icl', 'natural']
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    for idx, data_type in enumerate(data_types):
        ax = axes[idx]
        subset = df[df['data_type'] == data_type]
        
        if not subset.empty:
            # Pivot for heatmap-like visualization
            pivot_data = subset.pivot(index='layer', columns='cycle', values='entropy')
            
            # Create a simple heatmap using imshow
            im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index)
            
            ax.set_title(f'Entropy by Layer and Cycle: {data_type.replace("_", " ").title()}')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Layer')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Entropy (bits)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    """Main function to run entropy analysis."""
    
    # Configuration
    base_path = "/home/mmahaut/projects/parrots/outputs_multihead_full/EleutherAI/pythia-1.4b/steplatest"
    layers = list(range(24))  # All 24 layers
    cycles = list(range(5))   # Cycles 0-4
    
    print("Starting entropy analysis for repetitive cycles...")
    print(f"Analyzing {len(layers)} layers across {len(cycles)} cycles")
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    # Analyze entropy across cycles
    print("Loading and analyzing cycle data...")
    df = analyze_entropy_across_cycles(base_path, layers, cycles)
    
    if df.empty:
        print("No data found. Please check the data paths and format.")
        return
    
    print(f"Successfully loaded data for {len(df)} data points")
    print(f"Data types found: {df['data_type'].unique()}")
    print(f"Layers analyzed: {sorted(df['layer'].unique())}")
    print(f"Cycles analyzed: {sorted(df['cycle'].unique())}")
    
    # Generate plots
    print("Generating entropy evolution plot...")
    plot_entropy_evolution(df, "/home/mmahaut/projects/parrots/entropy_evolution_across_cycles.png")
    
    print("Generating entropy by layer plot...")
    plot_entropy_by_layer(df, "/home/mmahaut/projects/parrots/entropy_by_layer_and_cycle.png")
    
    # Save summary statistics
    summary_path = "/home/mmahaut/projects/parrots/entropy_analysis_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Summary data saved to: {summary_path}")
    
    # Print basic statistics
    print("\n=== ENTROPY ANALYSIS SUMMARY ===")
    for data_type in df['data_type'].unique():
        subset = df[df['data_type'] == data_type]
        print(f"\n{data_type.replace('_', ' ').title()}:")
        print(f"  Mean entropy: {subset['entropy'].mean():.3f}")
        print(f"  Std entropy: {subset['entropy'].std():.3f}")
        print(f"  Min entropy: {subset['entropy'].min():.3f}")
        print(f"  Max entropy: {subset['entropy'].max():.3f}")
    
    print("\n=== Analysis completed! ===")


if __name__ == "__main__":
    main()