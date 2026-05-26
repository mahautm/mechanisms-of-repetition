#!/usr/bin/env python3
"""
Create Publication-Ready Comparison Figure: Natural vs No-Cycle-ICL
===================================================================

Generates a clear comparison showing:
1. Natural (repetitive) vs No-Cycle-ICL (non-repetitive) attention shifts
2. Examples of token types with annotations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    """Calculate attention shifts for both Natural and No-Cycle-ICL."""
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
                natural_baseline[token_type].append(attention * 100)
            
            for token_type, attention in no_newline_dist.items():
                if token_type not in natural_no_newline:
                    natural_no_newline[token_type] = []
                natural_no_newline[token_type].append(attention * 100)
    
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
    
    # Merge TEMPLATE_WORD and SEMANTIC_ENTITY into CONTENT_WORD
    # (template words should be considered as content words)
    for token_type in ['TEMPLATE_WORD', 'SEMANTIC_ENTITY']:
        if token_type in natural_shifts:
            if 'CONTENT_WORD' not in natural_shifts:
                natural_shifts['CONTENT_WORD'] = 0
            natural_shifts['CONTENT_WORD'] = natural_shifts['CONTENT_WORD'] + natural_shifts[token_type]
            del natural_shifts[token_type]
    
    for token_type in ['TEMPLATE_WORD', 'SEMANTIC_ENTITY']:
        if token_type in no_cycle_shifts:
            if 'CONTENT_WORD' not in no_cycle_shifts:
                no_cycle_shifts['CONTENT_WORD'] = 0
            no_cycle_shifts['CONTENT_WORD'] = no_cycle_shifts['CONTENT_WORD'] + no_cycle_shifts[token_type]
            del no_cycle_shifts[token_type]
    
    return natural_shifts, no_cycle_shifts

def create_comparison_figure(natural_shifts, no_cycle_shifts, output_path):
    """Create side-by-side comparison of Natural vs ICL."""
    
    # Token type definitions with examples
    token_examples = {
        'OTHER': 'BOS, special tokens',
        'CONTENT_WORD': '·word, ·text, Hello, USA',  # Nouns, verbs, adjectives
        'FUNCTION_WORD': '·the, ·and, ·with, ·of',  # Determiners, prepositions, conjunctions
        'PROGRAMMING': 'def, class, import, =',  # Programming tokens
        'SENTENCE_END': '.  !  ?',
        'PUNCTUATION': ',  ;  :',
        'NUMBER': '42, 3.14',
        'BRACKET': '(  )  [  ]',
    }
    
    # Define single colors for Natural and ICL
    natural_color = '#e74c3c'  # Red for Natural
    icl_color = '#3498db'  # Blue for ICL
    
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
                   label='ICL', alpha=0.85, color=icl_color,
                   edgecolor='black', linewidth=0.8)
    
    # Customize
    ax.set_ylabel('Attention Shift (pp)', fontweight='bold')
    ax.set_xlabel('Token Type', fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Create labels without examples - cleaner format
    labels = []
    for tt in filtered_types:
        # Shorten label names
        short_label = {
            'CONTENT_WORD': 'Content Word',
            'FUNCTION_WORD': 'Function',
            'TEMPLATE_WORD': 'Template Word',
            'SEMANTIC_FUNCTION': 'Semantic Function',
            'SEMANTIC_ENTITY': 'Named Entity',
            'PROGRAMMING': 'Programming',
            'SENTENCE_END': 'Sentence End',
            'PUNCTUATION': 'Punctuation',
            'NUMBER': 'Number',
            'BRACKET': 'Bracket',
            'OTHER': 'Other'
        }.get(tt, tt)
        
        labels.append(short_label)
    
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    
    # Horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
    
    # Add value labels on bars (only for significant differences)
    for i, (nat_val, no_cyc_val) in enumerate(zip(natural_values, no_cycle_values)):
        diff = abs(nat_val - no_cyc_val)
        if diff > 1.0:  # Only label if there's a significant difference
            # Add difference indicator
            max_val = max(nat_val, no_cyc_val)
            min_val = min(nat_val, no_cyc_val)
            y_pos = max_val + 1
            
            ax.plot([i - width/2, i + width/2], [y_pos, y_pos], 'k-', linewidth=1.5)
            ax.plot([i - width/2, i - width/2], [y_pos - 0.5, y_pos], 'k-', linewidth=1.5)
            ax.plot([i + width/2, i + width/2], [y_pos - 0.5, y_pos], 'k-', linewidth=1.5)
            ax.text(i, y_pos + 1, f'Δ{diff:.1f}', ha='center', va='bottom', 
                   fontsize=7, fontweight='bold', style='italic')
    
    # Legend - simple with just Natural and ICL
    legend_elements = [
        Patch(facecolor=natural_color, edgecolor='black', linewidth=0.8, 
              label='Natural', alpha=0.85),
        Patch(facecolor=icl_color, edgecolor='black', linewidth=0.8, 
              label='ICL', alpha=0.85),
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
    print(f"✅ Comparison figure saved: {output_path}")
    plt.close()
    
    return filtered_types, natural_values, no_cycle_values

def print_summary(natural_shifts, no_cycle_shifts, filtered_types):
    """Print summary statistics."""
    print("\n📊 Summary Statistics:")
    print("\n" + "="*70)
    print(f"{'Token Type':<20} {'Natural (pp)':<15} {'ICL (pp)':<15} {'Difference':<15}")
    print("="*70)
    
    for tt in filtered_types:
        nat_val = natural_shifts.get(tt, 0)
        no_cyc_val = no_cycle_shifts.get(tt, 0)
        diff = nat_val - no_cyc_val
        
        symbol = "⚠️ " if abs(diff) > 5 else "  "
        print(f"{symbol}{tt:<18} {nat_val:>+8.2f}        {no_cyc_val:>+8.2f}        {diff:>+8.2f}")
    
    print("="*70)
    
    # Category totals
    semantic_types = ['CONTENT_WORD', 'FUNCTION_WORD', 'SEMANTIC_FUNCTION']  # All word types
    structural_types = ['SENTENCE_END', 'PUNCTUATION', 'BRACKET']
    
    nat_semantic = sum(natural_shifts.get(t, 0) for t in semantic_types)
    nat_structural = sum(natural_shifts.get(t, 0) for t in structural_types)
    no_cyc_semantic = sum(no_cycle_shifts.get(t, 0) for t in semantic_types)
    no_cyc_structural = sum(no_cycle_shifts.get(t, 0) for t in structural_types)
    
    print(f"\n{'Category':<20} {'Natural (pp)':<15} {'ICL (pp)':<15} {'Difference':<15}")
    print("-"*70)
    print(f"{'Semantic Tokens':<20} {nat_semantic:>+8.2f}        {no_cyc_semantic:>+8.2f}        {nat_semantic - no_cyc_semantic:>+8.2f}")
    print(f"{'Structural Tokens':<20} {nat_structural:>+8.2f}        {no_cyc_structural:>+8.2f}        {nat_structural - no_cyc_structural:>+8.2f}")
    print("="*70)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create publication-ready comparison figures")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1.4b", help="Model name for results")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing results JSON")
    args = parser.parse_args()
    
    # Derive safe model name for paths
    safe_model_name = args.model_name.replace("/", "_")
    
    # Paths
    if args.results_dir:
        results_json = Path(args.results_dir) / "attention_fallback_comparison_results.json"
        output_dir = Path(args.results_dir)
    else:
        results_json = Path(f"/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison_{safe_model_name}/attention_fallback_comparison_results.json")
        output_dir = Path(f"/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison_{safe_model_name}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_comparison_path = output_dir / f"paper_figure_natural_vs_icl_clean_{safe_model_name}.png"
    
    print("📊 Creating Natural vs ICL comparison figure...")
    print(f"📂 Loading results from: {results_json}")
    
    # Load data
    results = load_results(results_json)
    
    # Calculate shifts for both sequence types
    natural_shifts, no_cycle_shifts = calculate_attention_shifts(results)
    
    print(f"\n📈 Attention shifts calculated:")
    print(f"   Natural sequences: {len([r for r in results['natural_results'] if r['success']])} analyzed")
    print(f"   ICL sequences: {len([r for r in results['no_cycle_results'] if r['success']])} analyzed")
    
    # Create comparison figure
    print(f"\n🎨 Creating comparison figure...")
    filtered_types, natural_values, no_cycle_values = create_comparison_figure(
        natural_shifts, no_cycle_shifts, output_comparison_path
    )
    
    # Print summary
    print_summary(natural_shifts, no_cycle_shifts, filtered_types)
    
    print(f"\n✅ Figure created successfully for {args.model_name}!")
    print(f"📁 Saved to: {output_comparison_path}")

if __name__ == "__main__":
    main()
