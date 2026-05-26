#!/usr/bin/env python3
"""
Create Publication-Ready Baseline Attention Distribution Figure
================================================================

Generates a clean, paper-ready version of the baseline attention distribution
showing where attention    # Format labels
    label_map = {
        'CONTENT_WORD': 'Content\\nWord',
        'TEMPLATE_WORD': 'Template\\nWord',
        'SEMANTIC_FUNCTION': 'Semantic\\nFunction',
        'SENTENCE_END': 'Sentence\\nEnd',
        'PUNCTUATION': 'Punct.',
        'BRACKET': 'Bracket',
        'WHITESPACE': 'Space',
        'NUMBER': 'Number',
        'OTHER': 'Other',
        'NEWLINE': 'Newline'
    }lines are removed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,
})

def load_results(json_path: Path):
    """Load analysis results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_attention_shifts(results):
    """Calculate attention shifts from baseline to no-newline condition for both Natural and No-Cycle-ICL."""
    natural_baseline = {}
    natural_no_newline = {}
    no_cycle_baseline = {}
    no_cycle_no_newline = {}
    
    # Aggregate Natural sequences
    for sequence in results.get('natural_results', []):
        if not sequence.get('success', False):
            continue
        
        for head_result in sequence.get('head_results', []):
            baseline_dist = head_result.get('baseline_distribution', {})
            no_newline_dist = head_result.get('no_newline_distribution', {})
            
            for token_type, attention in baseline_dist.items():
                if token_type not in natural_baseline:
                    natural_baseline[token_type] = []
                natural_baseline[token_type].append(attention * 100)  # Convert to percentage
            
            for token_type, attention in no_newline_dist.items():
                if token_type not in natural_no_newline:
                    natural_no_newline[token_type] = []
                natural_no_newline[token_type].append(attention * 100)  # Convert to percentage
    
    # Aggregate No-Cycle-ICL sequences
    for sequence in results.get('no_cycle_results', []):
        if not sequence.get('success', False):
            continue
        
        for head_result in sequence.get('head_results', []):
            baseline_dist = head_result.get('baseline_distribution', {})
            no_newline_dist = head_result.get('no_newline_distribution', {})
            
            for token_type, attention in baseline_dist.items():
                if token_type not in no_cycle_baseline:
                    no_cycle_baseline[token_type] = []
                no_cycle_baseline[token_type].append(attention * 100)
            
            for token_type, attention in no_newline_dist.items():
                if token_type not in no_cycle_no_newline:
                    no_cycle_no_newline[token_type] = []
                no_cycle_no_newline[token_type].append(attention * 100)
    
    # Calculate mean shifts for both
    natural_shifts = {}
    no_cycle_shifts = {}
    all_token_types = set(list(natural_baseline.keys()) + list(natural_no_newline.keys()) + 
                          list(no_cycle_baseline.keys()) + list(no_cycle_no_newline.keys()))
    
    for token_type in all_token_types:
        natural_baseline_mean = np.mean(natural_baseline.get(token_type, [0]))
        natural_no_newline_mean = np.mean(natural_no_newline.get(token_type, [0]))
        natural_shifts[token_type] = natural_no_newline_mean - natural_baseline_mean
        
        no_cycle_baseline_mean = np.mean(no_cycle_baseline.get(token_type, [0]))
        no_cycle_no_newline_mean = np.mean(no_cycle_no_newline.get(token_type, [0]))
        no_cycle_shifts[token_type] = no_cycle_no_newline_mean - no_cycle_baseline_mean
    
    # Merge SEMANTIC_ENTITY and TEMPLATE_WORD (both are high-frequency template words)
    if 'SEMANTIC_ENTITY' in natural_shifts and 'TEMPLATE_WORD' in natural_shifts:
        natural_shifts['TEMPLATE_WORD'] = natural_shifts['TEMPLATE_WORD'] + natural_shifts['SEMANTIC_ENTITY']
        del natural_shifts['SEMANTIC_ENTITY']
    elif 'SEMANTIC_ENTITY' in natural_shifts:
        natural_shifts['TEMPLATE_WORD'] = natural_shifts['SEMANTIC_ENTITY']
        del natural_shifts['SEMANTIC_ENTITY']
    
    if 'SEMANTIC_ENTITY' in no_cycle_shifts and 'TEMPLATE_WORD' in no_cycle_shifts:
        no_cycle_shifts['TEMPLATE_WORD'] = no_cycle_shifts['TEMPLATE_WORD'] + no_cycle_shifts['SEMANTIC_ENTITY']
        del no_cycle_shifts['SEMANTIC_ENTITY']
    elif 'SEMANTIC_ENTITY' in no_cycle_shifts:
        no_cycle_shifts['TEMPLATE_WORD'] = no_cycle_shifts['SEMANTIC_ENTITY']
        del no_cycle_shifts['SEMANTIC_ENTITY']
    
    return natural_shifts, no_cycle_shifts, natural_baseline, natural_no_newline

def create_comparison_figure(natural_shifts, no_cycle_shifts, output_path):
    """Create publication-ready figure comparing Natural vs No-Cycle-ICL attention shifts."""
    
    # Define token type categories and colors
    structural_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'BRACKET', 'WHITESPACE']
    semantic_types = ['CONTENT_WORD', 'TEMPLATE_WORD', 'SEMANTIC_FUNCTION']
    other_types = ['NUMBER', 'OTHER']
    
    # Get all token types and filter significant ones (excluding NEWLINE which always decreases)
    all_types = set(list(natural_shifts.keys()) + list(no_cycle_shifts.keys()))
    filtered_types = [tt for tt in all_types if tt != 'NEWLINE' and 
                     (abs(natural_shifts.get(tt, 0)) > 0.005 or abs(no_cycle_shifts.get(tt, 0)) > 0.005)]
    
    # Sort by average absolute shift magnitude
    filtered_types = sorted(filtered_types, 
                           key=lambda x: abs(natural_shifts.get(x, 0)) + abs(no_cycle_shifts.get(x, 0)), 
                           reverse=True)[:8]  # Top 8 most significant
    
    # Assign colors based on category
    colors = []
    for tt in filtered_types:
        if tt in structural_types:
            colors.append('#e74c3c')  # Red for structural
        elif tt in semantic_types:
            colors.append('#3498db')  # Blue for semantic
        else:
            colors.append('#95a5a6')  # Gray for other
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    
    shift_values = [shifts[tt] for tt in filtered_types]
    x_pos = np.arange(len(filtered_types))
    
    # Create bars
    bars = ax.bar(x_pos, shift_values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # Customize appearance
    ax.set_ylabel('Attention Shift (pp)', fontweight='bold')
    ax.set_xlabel('Token Type', fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Format x-axis labels
    label_map = {
        'CONTENT_WORD': 'Content\nWord',
        'TEMPLATE_WORD': 'Template\nWord',
        'SEMANTIC_FUNCTION': 'Semantic\nFunction',
        'SENTENCE_END': 'Sentence\nEnd',
        'PUNCTUATION': 'Punct.',
        'BRACKET': 'Bracket',
        'WHITESPACE': 'Space',
        'NUMBER': 'Number',
        'OTHER': 'Other'
    }
    
    ax.set_xticklabels([label_map.get(tt, tt) for tt in filtered_types], 
                       rotation=0, ha='center', fontsize=8)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, shift_values)):
        if abs(val) > 0.01:  # Only label significant values
            label = f'{val:.2f}'
            y_offset = 0.02 if val > 0 else -0.06
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_offset,
                   label, ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=7, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', linewidth=0.8, label='Semantic'),
        Patch(facecolor='#e74c3c', edgecolor='black', linewidth=0.8, label='Structural'),
        Patch(facecolor='#95a5a6', edgecolor='black', linewidth=0.8, label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
             edgecolor='black', fancybox=False)
    
    # Grid
    ax.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis limits for better visualization
    max_abs_shift = max(abs(min(shift_values)), abs(max(shift_values)))
    ax.set_ylim(-max_abs_shift * 1.2, max_abs_shift * 1.2)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Paper-ready figure saved: {output_path}")
    plt.close()
    
    return filtered_types, shift_values

def create_baseline_distribution_figure(baseline_data, output_path):
    """Create figure showing baseline attention distribution (with newlines)."""
    
    # Calculate mean baseline attention for each token type
    baseline_means = {}
    baseline_stds = {}
    
    for token_type, values in baseline_data.items():
        if len(values) > 0:
            baseline_means[token_type] = np.mean(values)
            baseline_stds[token_type] = np.std(values)
    
    # Sort by mean attention
    sorted_types = sorted(baseline_means.keys(), key=lambda x: baseline_means[x], reverse=True)
    
    # Filter types with attention > 0.1%
    filtered_types = [tt for tt in sorted_types if baseline_means[tt] > 0.1]
    
    # Define colors
    structural_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'BRACKET', 'WHITESPACE']
    semantic_types = ['CONTENT_WORD', 'TEMPLATE_WORD', 'SEMANTIC_FUNCTION']
    
    colors = []
    for tt in filtered_types:
        if tt in structural_types:
            colors.append('#e74c3c')
        elif tt in semantic_types:
            colors.append('#3498db')
        else:
            colors.append('#95a5a6')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    
    means = [baseline_means[tt] for tt in filtered_types]
    stds = [baseline_stds[tt] for tt in filtered_types]
    x_pos = np.arange(len(filtered_types))
    
    # Create bars with error bars
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=0.8, capsize=3, error_kw={'linewidth': 1})
    
    # Customize
    ax.set_ylabel('Attention (%)', fontweight='bold')
    ax.set_xlabel('Token Type', fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Format labels
    label_map = {
        'CONTENT_WORD': 'Content\nWord',
        'TEMPLATE_WORD': 'Template\nWord',
        'SEMANTIC_FUNCTION': 'Semantic\nFunction',
        'SEMANTIC_ENTITY': 'Named\nEntity',
        'SENTENCE_END': 'Sentence\nEnd',
        'PUNCTUATION': 'Punct.',
        'BRACKET': 'Bracket',
        'WHITESPACE': 'Space',
        'NUMBER': 'Number',
        'OTHER': 'Other',
        'NEWLINE': 'Newline'
    }
    
    ax.set_xticklabels([label_map.get(tt, tt) for tt in filtered_types],
                       rotation=0, ha='center', fontsize=8)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        if mean > 1:  # Only label significant values
            label = f'{mean:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 1,
                   label, ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', linewidth=0.8, label='Semantic'),
        Patch(facecolor='#e74c3c', edgecolor='black', linewidth=0.8, label='Structural'),
        Patch(facecolor='#95a5a6', edgecolor='black', linewidth=0.8, label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
             edgecolor='black', fancybox=False)
    
    # Grid
    ax.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Baseline distribution figure saved: {output_path}")
    plt.close()

def main():
    # Paths
    results_json = Path("/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison/attention_fallback_comparison_results.json")
    output_dir = Path("/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison")
    
    output_shift_path = output_dir / "paper_figure_attention_shifts.png"
    output_baseline_path = output_dir / "paper_figure_baseline_distribution.png"
    
    print("📊 Creating paper-ready figures...")
    print(f"📂 Loading results from: {results_json}")
    
    # Load data
    results = load_results(results_json)
    
    # Calculate shifts
    shifts, baseline_data, no_newline_data = calculate_attention_shifts(results)
    
    print(f"\n📈 Calculated attention shifts for {len(shifts)} token types")
    print(f"   Top 3 increases: ", end="")
    top_increases = sorted(shifts.items(), key=lambda x: x[1], reverse=True)[:3]
    print(", ".join([f"{tt}: +{val:.2f}pp" for tt, val in top_increases]))
    
    # Create figures
    print(f"\n🎨 Creating attention shift figure...")
    filtered_types, shift_values = create_paper_figure(shifts, baseline_data, output_shift_path)
    
    print(f"\n🎨 Creating baseline distribution figure...")
    create_baseline_distribution_figure(baseline_data, output_baseline_path)
    
    print(f"\n✅ All figures created successfully!")
    print(f"   1. Attention shifts: {output_shift_path}")
    print(f"   2. Baseline distribution: {output_baseline_path}")

if __name__ == "__main__":
    main()
