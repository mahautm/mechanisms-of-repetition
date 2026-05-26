#!/usr/bin/env python3
"""
Create comprehensive analysis showing all heads' token specializations for publication.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict, Counter
import argparse
import pandas as pd

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

print("✅ All imports successful!")

def classify_token_type(token):
    """Classify tokens into meaningful categories for analysis."""
    if token in ['Ċ', 'čĊ']:
        return 'NEWLINE'
    elif token in ['.', '!', '?']:
        return 'SENTENCE_END'
    elif token in [',', ';', ':']:
        return 'PUNCTUATION'
    elif token in ['The', 'Hello', 'Python', 'Machine', 'Att']:
        return 'TEMPLATE_WORD'
    elif token.startswith('Ġ'):  # GPT-style space prefix
        return 'CONTENT_WORD'
    elif token in ['{', '}', '(', ')', '[', ']']:
        return 'BRACKET'
    elif token.isdigit() or any(c.isdigit() for c in token):
        return 'NUMBER'
    else:
        return 'OTHER'

def load_cycle_evolution_data(data_path):
    """Load existing cycle evolution data from .pt files."""
    print(f"📥 Loading cycle evolution data from: {data_path}")
    
    all_data = {}
    
    for pt_file in Path(data_path).glob("cycle_evolution_parametric_c4_l*_all_results.pt"):
        try:
            layer_match = re.search(r'_l(\d+)_', pt_file.name)
            if not layer_match:
                continue
            layer_num = int(layer_match.group(1))
            
            data = torch.load(pt_file, map_location='cpu')
            all_data[layer_num] = data
            
        except Exception as e:
            print(f"   ❌ Error loading {pt_file}: {e}")
    
    return all_data

def analyze_all_heads_specialization(cycle_data, seq_types=['natural', 'no_cycle_icl']):
    """Analyze ALL heads' specializations across all layers."""
    
    all_heads_data = {}
    
    # Get all possible heads from the data
    all_heads = []
    for layer_num in sorted(cycle_data.keys()):
        for seq_type in seq_types:
            layer_data = cycle_data.get(layer_num, {}).get(seq_type, {})
            focus_data = layer_data.get('focus_tokens', [])
            
            if focus_data:
                num_heads = len(focus_data[0]) if focus_data else 0
                for head_idx in range(num_heads):
                    head_id = (layer_num, head_idx)
                    if head_id not in all_heads:
                        all_heads.append(head_id)
    
    print(f"🔍 Found {len(all_heads)} total heads across {len(set(h[0] for h in all_heads))} layers")
    
    # Analyze each head
    for layer_num, head_idx in all_heads:
        all_heads_data[(layer_num, head_idx)] = {}
        
        for seq_type in seq_types:
            layer_data = cycle_data.get(layer_num, {}).get(seq_type, {})
            focus_data = layer_data.get('focus_tokens', [])
            
            if not focus_data or head_idx >= len(focus_data[0]) if focus_data else True:
                continue
            
            # Collect all tokens and categorize them
            token_categories = defaultdict(int)
            total_tokens = 0
            
            for seq_focus_tokens in focus_data:
                if head_idx < len(seq_focus_tokens):
                    head_tokens = seq_focus_tokens[head_idx]
                    for token_info in head_tokens:
                        token = token_info['token']
                        category = classify_token_type(token)
                        token_categories[category] += 1
                        total_tokens += 1
            
            # Calculate percentages
            category_percentages = {}
            for category in ['NEWLINE', 'TEMPLATE_WORD', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'BRACKET', 'NUMBER', 'OTHER']:
                count = token_categories.get(category, 0)
                category_percentages[category] = (count / total_tokens * 100) if total_tokens > 0 else 0
            
            all_heads_data[(layer_num, head_idx)][seq_type] = {
                'categories': category_percentages,
                'total_tokens': total_tokens,
                'dominant_category': max(category_percentages.items(), key=lambda x: x[1]) if category_percentages else ('UNKNOWN', 0)
            }
    
    return all_heads_data, sorted(all_heads)

def create_comprehensive_specialization_matrix(all_heads_data, all_heads, output_dir):
    """Create a comprehensive matrix showing all heads' specializations."""
    
    categories = ['NEWLINE', 'TEMPLATE_WORD', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'BRACKET', 'NUMBER', 'OTHER']
    seq_types = ['natural', 'no_cycle_icl']
    
    # Create figure with subplots for each sequence type
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    
    for seq_idx, seq_type in enumerate(seq_types):
        ax = axes[seq_idx]
        
        # Prepare data matrix
        matrix_data = []
        head_labels = []
        
        for layer_num, head_idx in all_heads:
            head_label = f'L{layer_num}H{head_idx}'
            head_labels.append(head_label)
            
            head_data = all_heads_data.get((layer_num, head_idx), {})
            seq_data = head_data.get(seq_type, {})
            categories_data = seq_data.get('categories', {})
            
            row = [categories_data.get(cat, 0) for cat in categories]
            matrix_data.append(row)
        
        # Create heatmap
        im = ax.imshow(matrix_data, cmap='viridis', aspect='auto', vmin=0, vmax=100)
        
        # Set labels and title
        ax.set_title(f'{seq_type.replace("_", " ").title()} Sequences - Token Focus Distribution (%)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Token Categories', fontsize=14)
        ax.set_ylabel('Attention Heads (Layer-Head)', fontsize=14)
        
        # Set ticks
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # Only show every 10th head label to avoid crowding
        head_tick_indices = range(0, len(head_labels), 10)
        ax.set_yticks(head_tick_indices)
        ax.set_yticklabels([head_labels[i] for i in head_tick_indices])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Focus Percentage (%)', rotation=270, labelpad=20)
        
        # Add grid for better readability
        ax.set_xticks(np.arange(len(categories)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(head_labels)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    matrix_path = output_dir / "comprehensive_head_specialization_matrix.png"
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Comprehensive matrix saved: {matrix_path}")

def create_specialization_summary_by_layer(all_heads_data, all_heads, output_dir):
    """Create layer-wise summary of specializations."""
    
    categories = ['NEWLINE', 'TEMPLATE_WORD', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'OTHER']
    seq_types = ['natural', 'no_cycle_icl']
    
    # Group heads by layer
    layer_groups = defaultdict(list)
    for layer_num, head_idx in all_heads:
        layer_groups[layer_num].append(head_idx)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for seq_idx, seq_type in enumerate(seq_types):
        ax = axes[seq_idx]
        
        # Prepare data for each layer
        layer_nums = sorted(layer_groups.keys())
        layer_means = {cat: [] for cat in categories}
        layer_stds = {cat: [] for cat in categories}
        
        for layer_num in layer_nums:
            heads_in_layer = layer_groups[layer_num]
            
            # Calculate mean and std for each category in this layer
            for category in categories:
                values = []
                for head_idx in heads_in_layer:
                    head_data = all_heads_data.get((layer_num, head_idx), {})
                    seq_data = head_data.get(seq_type, {})
                    categories_data = seq_data.get('categories', {})
                    values.append(categories_data.get(category, 0))
                
                layer_means[category].append(np.mean(values) if values else 0)
                layer_stds[category].append(np.std(values) if len(values) > 1 else 0)
        
        # Create stacked area plot or line plot
        colors = {
            'NEWLINE': '#e74c3c',
            'TEMPLATE_WORD': '#3498db',
            'SENTENCE_END': '#2ecc71',
            'PUNCTUATION': '#f39c12',
            'CONTENT_WORD': '#9b59b6',
            'OTHER': '#95a5a6'
        }
        
        # Plot lines with error bars
        for category in categories:
            means = layer_means[category]
            stds = layer_stds[category]
            ax.errorbar(layer_nums, means, yerr=stds, label=category, 
                       color=colors.get(category, '#95a5a6'), 
                       marker='o', linewidth=2, capsize=3, alpha=0.8)
        
        ax.set_title(f'{seq_type.replace("_", " ").title()} Sequences - Average Specialization by Layer', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer Number', fontsize=12)
        ax.set_ylabel('Average Focus Percentage (%)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    summary_path = output_dir / "specialization_by_layer_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Layer summary saved: {summary_path}")

def create_average_focus_plot(all_heads_data, all_heads, output_dir):
    """Create average focus plot showing mean focus percentages across all heads."""
    
    seq_types = ['natural', 'no_cycle_icl']
    categories = ['NEWLINE', 'TEMPLATE_WORD', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'BRACKET', 'NUMBER', 'OTHER']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for seq_idx, seq_type in enumerate(seq_types):
        ax = axes[seq_idx]
        
        # Calculate average focus and statistics for each category
        category_stats = {}
        
        for category in categories:
            focus_values = []
            
            for layer_num, head_idx in all_heads:
                head_data = all_heads_data.get((layer_num, head_idx), {})
                seq_data = head_data.get(seq_type, {})
                categories_data = seq_data.get('categories', {})
                
                if seq_data.get('total_tokens', 0) > 0:
                    focus_pct = categories_data.get(category, 0)
                    focus_values.append(focus_pct)
            
            if focus_values:
                category_stats[category] = {
                    'mean': np.mean(focus_values),
                    'std': np.std(focus_values),
                    'median': np.median(focus_values),
                    'q75': np.percentile(focus_values, 75),
                    'q25': np.percentile(focus_values, 25),
                    'max': np.max(focus_values),
                    'count': len(focus_values)
                }
            else:
                category_stats[category] = {
                    'mean': 0, 'std': 0, 'median': 0, 'q75': 0, 'q25': 0, 'max': 0, 'count': 0
                }
        
        # Create bar plot with error bars
        categories_sorted = sorted(categories, key=lambda x: category_stats[x]['mean'], reverse=True)
        means = [category_stats[cat]['mean'] for cat in categories_sorted]
        stds = [category_stats[cat]['std'] for cat in categories_sorted]
        medians = [category_stats[cat]['median'] for cat in categories_sorted]
        maxes = [category_stats[cat]['max'] for cat in categories_sorted]
        
        colors = {
            'NEWLINE': '#e74c3c',
            'TEMPLATE_WORD': '#3498db',
            'SENTENCE_END': '#2ecc71',
            'PUNCTUATION': '#f39c12',
            'CONTENT_WORD': '#9b59b6',
            'BRACKET': '#34495e',
            'NUMBER': '#f1c40f',
            'OTHER': '#95a5a6'
        }
        
        bar_colors = [colors.get(cat, '#95a5a6') for cat in categories_sorted]
        
        # Create bars for means with error bars
        bars = ax.bar(range(len(categories_sorted)), means, yerr=stds, 
                     color=bar_colors, alpha=0.7, capsize=5, 
                     error_kw={'linewidth': 2, 'capthick': 2})
        
        # Add median markers
        ax.scatter(range(len(categories_sorted)), medians, 
                  color='white', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=2, label='Median')
        
        # Add max value markers
        ax.scatter(range(len(categories_sorted)), maxes, 
                  color='black', s=60, zorder=5, marker='^', 
                  label='Maximum')
        
        # Add value labels on bars
        for i, (mean, std, median, max_val) in enumerate(zip(means, stds, medians, maxes)):
            # Mean value
            ax.text(i, mean + std + 1, f'{mean:.1f}%', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
            # Max value in parentheses
            ax.text(i, max_val + 2, f'(max: {max_val:.0f}%)', 
                   ha='center', va='bottom', fontsize=8, style='italic')
        
        ax.set_title(f'{seq_type.replace("_", " ").title()} Sequences\nAverage Token Focus Across All Heads', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Token Categories', fontsize=12)
        ax.set_ylabel('Focus Percentage (%)', fontsize=12)
        
        # Set x-axis labels
        ax.set_xticks(range(len(categories_sorted)))
        ax.set_xticklabels(categories_sorted, rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Add grid for better readability
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max(maxes) * 1.1 if maxes else 1)
        
        # Add text box with statistics
        stats_text = f"Analyzed: {category_stats[categories_sorted[0]]['count']} heads"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
               verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    focus_path = output_dir / "average_focus_plot.png"
    plt.savefig(focus_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Average focus plot saved: {focus_path}")

def create_head_specialization_scatter(all_heads_data, all_heads, output_dir):
    """Create scatter plot showing head specialization patterns."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    seq_types = ['natural', 'no_cycle_icl']
    
    for seq_idx, seq_type in enumerate(seq_types):
        ax = axes[seq_idx]
        
        # Prepare data for scatter plot
        newline_focus = []
        template_focus = []
        layer_nums = []
        head_indices = []
        
        for layer_num, head_idx in all_heads:
            head_data = all_heads_data.get((layer_num, head_idx), {})
            seq_data = head_data.get(seq_type, {})
            categories_data = seq_data.get('categories', {})
            
            if seq_data.get('total_tokens', 0) > 0:
                newline_pct = categories_data.get('NEWLINE', 0)
                template_pct = categories_data.get('TEMPLATE_WORD', 0)
                
                newline_focus.append(newline_pct)
                template_focus.append(template_pct)
                layer_nums.append(layer_num)
                head_indices.append(head_idx)
        
        # Create scatter plot with layer-based coloring
        scatter = ax.scatter(newline_focus, template_focus, c=layer_nums, 
                           cmap='viridis', alpha=0.6, s=50)
        
        ax.set_xlabel('Newline Focus (%)', fontsize=12)
        ax.set_ylabel('Template Word Focus (%)', fontsize=12)
        ax.set_title(f'{seq_type.replace("_", " ").title()} Sequences\nNewline vs Template Specialization', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar for layer information
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Layer Number', rotation=270, labelpad=15)
        
        # Add quadrant lines
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(75, 75, 'High Both', fontsize=10, ha='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        ax.text(75, 10, 'Newline\nSpecialists', fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        ax.text(10, 75, 'Template\nSpecialists', fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))
        ax.text(10, 10, 'Generalists', fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    scatter_path = output_dir / "head_specialization_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Specialization scatter plot saved: {scatter_path}")

def main():
    parser = argparse.ArgumentParser(description="Create comprehensive analysis for all heads")
    parser.add_argument("--cycle_data_path", type=str,
                       default="/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/cycle_evolution_parametric/cycles_4/steplatest")
    parser.add_argument("--output_dir", type=str, default="./plots/comprehensive_analysis")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Creating comprehensive analysis for ALL heads...")
    
    # Load data
    cycle_data = load_cycle_evolution_data(args.cycle_data_path)
    
    # Analyze all heads
    print(f"🔬 Analyzing ALL heads' specializations...")
    all_heads_data, all_heads = analyze_all_heads_specialization(cycle_data)
    
    # Create comprehensive visualizations
    print(f"📊 Creating publication-quality plots...")
    create_comprehensive_specialization_matrix(all_heads_data, all_heads, output_dir)
    create_specialization_summary_by_layer(all_heads_data, all_heads, output_dir)
    create_average_focus_plot(all_heads_data, all_heads, output_dir)
    create_head_specialization_scatter(all_heads_data, all_heads, output_dir)
    
    print(f"\n✅ Comprehensive analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🎯 Analyzed {len(all_heads)} total heads across {len(set(h[0] for h in all_heads))} layers")

if __name__ == "__main__":
    main()