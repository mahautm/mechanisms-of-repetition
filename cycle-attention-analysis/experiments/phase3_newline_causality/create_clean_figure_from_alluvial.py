#!/usr/bin/env python3
"""
Create Clean Single-Plot Figure from Alluvial Results
=====================================================

Takes existing alluvial-style results and creates a clean publication figure.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import argparse

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

def load_alluvial_results(json_path: Path):
    """Load analysis results from alluvial-style JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_attention_shifts(results):
    """Calculate attention shifts for both Natural and No-Cycle-ICL."""
    natural_baseline = {}
    natural_no_newline = {}
    no_cycle_baseline = {}
    no_cycle_no_newline = {}
    
    # Aggregate Natural sequences
    for sequence in results.get('natural_results', []):
        if not sequence.get('success', False):
            continue
        
        # Handle alluvial-style structure with head_results
        head_results = sequence.get('head_results', [])
        if head_results:
            for head_result in head_results:
                baseline_dist = head_result.get('baseline_distribution', {})
                no_newline_dist = head_result.get('no_newline_distribution', {})
                
                for token_type, attention in baseline_dist.items():
                    if token_type not in natural_baseline:
                        natural_baseline[token_type] = []
                    natural_baseline[token_type].append(attention * 100)
                
                for token_type, attention in no_newline_dist.items():
                    if token_type not in natural_no_newline:
                        natural_no_newline[token_type] = []
                    natural_no_newline[token_type].append(attention * 100)
    
    # Aggregate No-Cycle-ICL sequences
    for sequence in results.get('no_cycle_results', []):
        if not sequence.get('success', False):
            continue
        
        # Handle alluvial-style structure with head_results
        head_results = sequence.get('head_results', [])
        if head_results:
            for head_result in head_results:
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
    
    # Calculate shifts
    natural_shifts = {}
    no_cycle_shifts = {}
    all_token_types = set(list(natural_baseline.keys()) + list(no_cycle_baseline.keys()))
    
    for token_type in all_token_types:
        natural_baseline_mean = np.mean(natural_baseline.get(token_type, [0]))
        natural_no_newline_mean = np.mean(natural_no_newline.get(token_type, [0]))
        natural_shifts[token_type] = natural_no_newline_mean - natural_baseline_mean
        
        no_cycle_baseline_mean = np.mean(no_cycle_baseline.get(token_type, [0]))
        no_cycle_no_newline_mean = np.mean(no_cycle_no_newline.get(token_type, [0]))
        no_cycle_shifts[token_type] = no_cycle_no_newline_mean - no_cycle_baseline_mean
    
    return natural_shifts, no_cycle_shifts

def create_clean_comparison_figure(natural_shifts, no_cycle_shifts, output_path, model_name="OLMo-1B"):
    """Create clean side-by-side comparison of Natural vs No-Cycle-ICL."""
    
    # Define single colors for Natural and No-Cycle-ICL
    natural_color = '#e74c3c'  # Red for Natural
    icl_color = '#3498db'  # Blue for No-Cycle-ICL
    
    # Filter significant token types (excluding NEWLINE)
    all_types = set(list(natural_shifts.keys()) + list(no_cycle_shifts.keys()))
    filtered_types = [tt for tt in all_types if tt != 'NEWLINE' and tt != 'WHITESPACE' and
                     (abs(natural_shifts.get(tt, 0)) > 0.01 or abs(no_cycle_shifts.get(tt, 0)) > 0.01)]
    
    # Sort by total magnitude
    filtered_types = sorted(filtered_types, 
                           key=lambda x: abs(natural_shifts.get(x, 0)) + abs(no_cycle_shifts.get(x, 0)), 
                           reverse=True)[:7]  # Top 7
    
    # Create figure with comparison
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    
    x_pos = np.arange(len(filtered_types))
    width = 0.38
    
    natural_values = [natural_shifts.get(tt, 0) for tt in filtered_types]
    no_cycle_values = [no_cycle_shifts.get(tt, 0) for tt in filtered_types]
    
    # Create grouped bars
    bars1 = ax.bar(x_pos - width/2, natural_values, width, 
                   label='Natural', alpha=0.85, color=natural_color, 
                   edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos + width/2, no_cycle_values, width,
                   label='No-Cycle-ICL', alpha=0.85, color=icl_color,
                   edgecolor='black', linewidth=0.8)
    
    # Customize
    ax.set_ylabel('Attention Shift (pp)', fontweight='bold')
    ax.set_xlabel('Token Type', fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Create shorter labels
    labels = []
    for tt in filtered_types:
        short_label = {
            'CONTENT_WORD': 'Content\nWord',
            'FUNCTION_WORD': 'Function\nWord',
            'PROGRAMMING': 'Program',
            'SENTENCE_END': 'Sentence\nEnd',
            'PUNCTUATION': 'Punctuation',
            'NUMBER': 'Number',
            'BRACKET': 'Bracket',
            'OTHER': 'Other'
        }.get(tt, tt)
        labels.append(short_label)
    
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    
    # Horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
    
    # Add value labels on bars (only for significant differences)
    for i, (nat_val, no_cyc_val) in enumerate(zip(natural_values, no_cycle_values)):
        diff = abs(nat_val - no_cyc_val)
        if diff > 0.03:  # Only label if there's a significant difference
            # Add difference indicator
            max_val = max(nat_val, no_cyc_val)
            y_pos = max_val + 0.01
            
            ax.plot([i - width/2, i + width/2], [y_pos, y_pos], 'k-', linewidth=1.5)
            ax.plot([i - width/2, i - width/2], [y_pos - 0.005, y_pos], 'k-', linewidth=1.5)
            ax.plot([i + width/2, i + width/2], [y_pos - 0.005, y_pos], 'k-', linewidth=1.5)
            ax.text(i, y_pos + 0.01, f'Δ{diff:.2f}', ha='center', va='bottom', 
                   fontsize=7, fontweight='bold', style='italic')
    
    # Legend
    legend_elements = [
        Patch(facecolor=natural_color, edgecolor='black', linewidth=0.8, 
              label='Natural (Repetitive)', alpha=0.85),
        Patch(facecolor=icl_color, edgecolor='black', linewidth=0.8, 
              label='No-Cycle-ICL', alpha=0.85),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.98, 
             edgecolor='black', fancybox=False, ncol=1, fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Clean comparison figure saved: {output_path}")
    plt.close()
    
    return filtered_types, natural_values, no_cycle_values

def print_summary(natural_shifts, no_cycle_shifts, filtered_types):
    """Print summary statistics."""
    print("\n📊 Summary Statistics:")
    print("\n" + "="*75)
    print(f"{'Token Type':<20} {'Natural (pp)':<15} {'No-Cycle-ICL (pp)':<20} {'Difference':<15}")
    print("="*75)
    
    for tt in filtered_types:
        nat_val = natural_shifts.get(tt, 0)
        no_cyc_val = no_cycle_shifts.get(tt, 0)
        diff = nat_val - no_cyc_val
        
        symbol = "⚠️ " if abs(diff) > 0.05 else "  "
        print(f"{symbol}{tt:<18} {nat_val:>+8.3f}        {no_cyc_val:>+8.3f}            {diff:>+8.3f}")
    
    print("="*75)

def main():
    parser = argparse.ArgumentParser(description="Create clean figure from alluvial results")
    parser.add_argument("--results_dir", type=str, required=True, 
                       help="Directory containing attention_fallback_alluvial_results.json")
    parser.add_argument("--model_name", type=str, default="OLMo-1B",
                       help="Model name for labeling")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_json = results_dir / "attention_fallback_alluvial_results.json"
    
    if not results_json.exists():
        print(f"❌ Results file not found: {results_json}")
        return
    
    # Derive safe model name for output
    safe_model_name = args.model_name.replace("/", "_").replace("-", "_")
    output_path = results_dir / f"paper_figure_natural_vs_nocycle_clean_{safe_model_name}.png"
    
    print(f"📊 Creating clean Natural vs No-Cycle-ICL comparison figure...")
    print(f"📂 Loading results from: {results_json}")
    
    # Load data
    results = load_alluvial_results(results_json)
    
    # Calculate shifts
    natural_shifts, no_cycle_shifts = calculate_attention_shifts(results)
    
    natural_count = len([r for r in results['natural_results'] if r['success']])
    no_cycle_count = len([r for r in results['no_cycle_results'] if r['success']])
    
    print(f"\n📈 Attention shifts calculated:")
    print(f"   Natural sequences: {natural_count} analyzed")
    print(f"   No-Cycle-ICL sequences: {no_cycle_count} analyzed")
    
    # Create clean figure
    print(f"\n🎨 Creating clean comparison figure...")
    filtered_types, natural_values, no_cycle_values = create_clean_comparison_figure(
        natural_shifts, no_cycle_shifts, output_path, args.model_name
    )
    
    # Print summary
    print_summary(natural_shifts, no_cycle_shifts, filtered_types)
    
    print(f"\n✅ Figure created successfully!")
    print(f"📁 Saved to: {output_path}")

if __name__ == "__main__":
    main()
