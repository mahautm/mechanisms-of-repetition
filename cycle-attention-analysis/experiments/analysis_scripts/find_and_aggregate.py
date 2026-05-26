#!/usr/bin/env python3
"""
Aggregate Attention Analysis Results Across All Layers
=====================================================

This script finds, loads, and aggregates attention analysis results from all 24 transformer layers
(EleutherAI/pythia-1.4b) to create comprehensive cross-layer visualizations and summaries.

WHAT IT DOES:
- 🔍 Searches for attention_analysis_layer_*.pt files across multiple directories
- 📊 Aggregates attention entropy and consistency metrics from all layers (0-23)
- 🧠 Compares Natural (repetitive) vs ICL (non-repetitive) sequences across layers
- 📈 Creates multi-layer trend plots and heatmaps
- 💾 Exports aggregated data as CSV for further analysis

INPUT DATA:
- Individual layer results: attention_analysis_layer_X.pt files
- Expected structure: {natural: {head_statistics: {...}}, icl: {head_statistics: {...}}}
- Metrics extracted: mean_consistency, mean_entropy per attention head

OUTPUT VISUALIZATIONS:
- attention_analysis_summary.png: 4-panel comparison (entropy trends, distributions)
- attention_heatmaps.png: Layer×Head attention patterns  
- aggregated_results.csv: Raw data for all layer/head combinations

KEY RESEARCH QUESTIONS ANSWERED:
- Which layers show biggest differences between repetitive/non-repetitive sequences?
- How does attention entropy evolve across the 24 transformer layers?
- Which layer/head combinations are most important for repetition mechanisms?
- Are attention patterns consistent across the entire model architecture?

USAGE:
    python find_and_aggregate.py

Author: Matéo Mahaut + Claude Sonnet 4
Date: October 2025
Context: Part of entropy-based analysis of attention mechanisms in transformers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import glob

def find_all_results():
    """Find all result files regardless of their location."""
    
    print("=== Finding All Result Files ===")
    
    # Search in multiple possible locations
    search_paths = [
        "../data/results/*/attention_analysis_layer_*.pt",
        "../data/test_results/*/attention_analysis_layer_*.pt", 
        "../results/*/attention_analysis_layer_*.pt",
        "*/attention_analysis_layer_*.pt"
    ]
    
    all_files = []
    for pattern in search_paths:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    print(f"Found {len(all_files)} result files:")
    for f in sorted(all_files):
        print(f"  {f}")
    
    return all_files

def load_and_aggregate_from_files(result_files):
    """Load and aggregate results from found files."""
    
    if not result_files:
        print("No result files found!")
        return None
    
    print(f"\nLoading and aggregating {len(result_files)} files...")
    
    all_data = {
        'layers': [],
        'natural_consistency': [],
        'icl_consistency': [], 
        'natural_entropy': [],
        'icl_entropy': [],
        'head_numbers': []
    }
    
    for result_file in sorted(result_files):
        try:
            # Extract layer number from filename
            layer_num = int(Path(result_file).stem.split('_')[-1])
            
            print(f"Loading layer {layer_num} from {result_file}")
            
            results = torch.load(result_file, map_location='cpu')
            
            # Process natural and ICL results
            for seq_type in ['natural', 'icl']:
                if seq_type in results and results[seq_type] is not None:
                    seq_data = results[seq_type]
                    head_stats = seq_data.get('head_statistics', {})
                    
                    print(f"  {seq_type}: {len(head_stats)} layers with stats")
                    
                    # Process each layer's heads
                    for layer_key, layer_heads in head_stats.items():
                        for head_key, head_data in layer_heads.items():
                            if isinstance(head_data, dict):
                                consistency = head_data.get('mean_consistency', 0.0)
                                entropy = head_data.get('mean_entropy', 0.0)
                                
                                # Extract head number
                                head_num = int(head_key.split('_')[1]) if '_' in head_key else 0
                                
                                all_data['layers'].append(layer_num)
                                all_data['head_numbers'].append(head_num)
                                
                                if seq_type == 'natural':
                                    all_data['natural_consistency'].append(consistency)
                                    all_data['natural_entropy'].append(entropy)
                                    all_data['icl_consistency'].append(0.0)
                                    all_data['icl_entropy'].append(0.0)
                                else:
                                    all_data['icl_consistency'].append(consistency)
                                    all_data['icl_entropy'].append(entropy)
                                    all_data['natural_consistency'].append(0.0)
                                    all_data['natural_entropy'].append(0.0)
            
        except Exception as e:
            print(f"  Error loading {result_file}: {e}")
            continue
    
    if all_data['layers']:
        df = pd.DataFrame(all_data)
        print(f"\nLoaded data for {len(df)} head-layer combinations")
        print(f"Layers: {sorted(df['layers'].unique())}")
        print(f"Natural entropy range: {df['natural_entropy'].min():.4f} - {df['natural_entropy'].max():.4f}")
        print(f"ICL entropy range: {df['icl_entropy'].min():.4f} - {df['icl_entropy'].max():.4f}")
        return df
    else:
        print("No data could be loaded!")
        return None

def create_visualizations_from_df(df):
    """Create visualizations from the dataframe."""
    
    if df is None or len(df) == 0:
        print("No data to visualize!")
        return
    
    output_dir = Path("../plots/aggregated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating visualizations with {len(df)} data points...")
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create main analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Natural entropy by layer (line plot)
    natural_entropy = df[df['natural_entropy'] > 0].groupby('layers')['natural_entropy'].mean()
    if len(natural_entropy) > 0:
        axes[0, 0].plot(natural_entropy.index, natural_entropy.values, 'o-', label='Natural')
        axes[0, 0].set_title('Natural Sequences - Average Entropy by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ICL entropy by layer (line plot)
    icl_entropy = df[df['icl_entropy'] > 0].groupby('layers')['icl_entropy'].mean()
    if len(icl_entropy) > 0:
        axes[0, 1].plot(icl_entropy.index, icl_entropy.values, 'o-', color='orange', label='ICL')
        axes[0, 1].set_title('ICL Sequences - Average Entropy by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Entropy comparison
    if len(natural_entropy) > 0 and len(icl_entropy) > 0:
        common_layers = set(natural_entropy.index) & set(icl_entropy.index)
        if common_layers:
            common_layers = sorted(common_layers)
            natural_vals = [natural_entropy[l] for l in common_layers]
            icl_vals = [icl_entropy[l] for l in common_layers]
            
            axes[1, 0].plot(common_layers, natural_vals, 'o-', label='Natural', alpha=0.7)
            axes[1, 0].plot(common_layers, icl_vals, 'o-', label='ICL', alpha=0.7)
            axes[1, 0].set_title('Entropy Comparison: Natural vs ICL')
            axes[1, 0].set_xlabel('Layer')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Entropy distribution
    entropy_data = []
    if df['natural_entropy'].sum() > 0:
        entropy_data.extend([('Natural', x) for x in df['natural_entropy'] if x > 0])
    if df['icl_entropy'].sum() > 0:
        entropy_data.extend([('ICL', x) for x in df['icl_entropy'] if x > 0])
    
    if entropy_data:
        entropy_df = pd.DataFrame(entropy_data, columns=['Type', 'Entropy'])
        sns.boxplot(data=entropy_df, x='Type', y='Entropy', ax=axes[1, 1])
        axes[1, 1].set_title('Entropy Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap if we have enough data
    if len(df) > 100:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Natural entropy heatmap
        if df['natural_entropy'].sum() > 0:
            pivot_natural = df[df['natural_entropy'] > 0].groupby(['layers', 'head_numbers'])['natural_entropy'].mean().unstack(fill_value=0)
            if not pivot_natural.empty:
                sns.heatmap(pivot_natural.T, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Entropy'})
                axes[0].set_title('Natural Sequences - Entropy by Layer and Head')
                axes[0].set_xlabel('Layer')
                axes[0].set_ylabel('Head')
        
        # ICL entropy heatmap
        if df['icl_entropy'].sum() > 0:
            pivot_icl = df[df['icl_entropy'] > 0].groupby(['layers', 'head_numbers'])['icl_entropy'].mean().unstack(fill_value=0)
            if not pivot_icl.empty:
                sns.heatmap(pivot_icl.T, ax=axes[1], cmap='viridis', cbar_kws={'label': 'Entropy'})
                axes[1].set_title('ICL Sequences - Entropy by Layer and Head')
                axes[1].set_xlabel('Layer')
                axes[1].set_ylabel('Head')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'attention_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save data
    df.to_csv(output_dir / 'aggregated_results.csv', index=False)
    
    print(f"✅ Visualizations saved to {output_dir}")
    print(f"   - attention_analysis_summary.png")
    print(f"   - attention_heatmaps.png") 
    print(f"   - aggregated_results.csv")

def main():
    """Find all results and aggregate them."""
    
    result_files = find_all_results()
    df = load_and_aggregate_from_files(result_files)
    create_visualizations_from_df(df)

if __name__ == "__main__":
    main()