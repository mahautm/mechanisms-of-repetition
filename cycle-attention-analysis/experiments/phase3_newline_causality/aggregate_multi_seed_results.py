#!/usr/bin/env python3
"""
Aggregate Multi-Seed Results for OLMo Attention Fallback Analysis
================================================================

This script aggregates results from multiple seed runs and creates
combined analysis plots showing data from all seeds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_all_seed_results(base_dir: Path, seeds: List[int]) -> Dict:
    """Load results from all seed directories."""
    all_results = {
        'natural_results': [],
        'no_cycle_results': [],
        'natural_shifts': defaultdict(list),
        'no_cycle_shifts': defaultdict(list),
        'seeds_loaded': []
    }
    
    for seed in seeds:
        result_dir = base_dir / f"attention_fallback_comparison_allenai_OLMo-1B-hf_seed{seed}"
        result_file = result_dir / "attention_fallback_comparison_results.json"
        
        if not result_file.exists():
            print(f"⚠️  Skipping seed {seed}: results not found")
            continue
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Aggregate results
            all_results['natural_results'].extend(data.get('natural_results', []))
            all_results['no_cycle_results'].extend(data.get('no_cycle_results', []))
            
            # Aggregate shifts
            for token_type, shift in data.get('natural_shifts', {}).items():
                all_results['natural_shifts'][token_type].append(shift)
            
            for token_type, shift in data.get('no_cycle_shifts', {}).items():
                all_results['no_cycle_shifts'][token_type].append(shift)
            
            all_results['seeds_loaded'].append(seed)
            print(f"✅ Loaded seed {seed}: {len(data.get('natural_results', []))} natural, {len(data.get('no_cycle_results', []))} no-cycle-icl")
            
        except Exception as e:
            print(f"⚠️  Error loading seed {seed}: {e}")
    
    return all_results

def calculate_aggregate_shifts(shifts_by_seed: Dict[str, List[float]]) -> Dict[str, Dict]:
    """Calculate mean, std, and confidence intervals for shifts."""
    aggregated = {}
    
    for token_type, values in shifts_by_seed.items():
        if len(values) == 0:
            continue
        
        aggregated[token_type] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_seeds': len(values)
        }
    
    return aggregated

def plot_aggregated_comparison(natural_agg: Dict, no_cycle_agg: Dict, output_dir: Path, n_seeds: int):
    """Create aggregated comparison plots with error bars."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Get all token types
    all_token_types = sorted(set(list(natural_agg.keys()) + list(no_cycle_agg.keys())))
    
    # 1. Attention Shifts with Error Bars
    ax1 = axes[0, 0]
    
    natural_means = [natural_agg.get(tt, {}).get('mean', 0) for tt in all_token_types]
    natural_stds = [natural_agg.get(tt, {}).get('std', 0) for tt in all_token_types]
    no_cycle_means = [no_cycle_agg.get(tt, {}).get('mean', 0) for tt in all_token_types]
    no_cycle_stds = [no_cycle_agg.get(tt, {}).get('std', 0) for tt in all_token_types]
    
    x_pos = np.arange(len(all_token_types))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, natural_means, width, yerr=natural_stds, 
                    label=f'Natural (n={n_seeds} seeds)', alpha=0.8, color='#e74c3c', capsize=5)
    bars2 = ax1.bar(x_pos + width/2, no_cycle_means, width, yerr=no_cycle_stds,
                    label=f'No-Cycle-ICL (n={n_seeds} seeds)', alpha=0.8, color='#3498db', capsize=5)
    
    ax1.set_xlabel('Token Type', fontsize=12)
    ax1.set_ylabel('Attention Shift (No-Newline - Baseline)', fontsize=12)
    ax1.set_title(f'🧠 Aggregated Attention Fallback (n={n_seeds} seeds)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_token_types, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars1, natural_means, natural_stds)):
        if abs(mean) > 0.001:
            ax1.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width() / 2, mean),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    for i, (bar, mean, std) in enumerate(zip(bars2, no_cycle_means, no_cycle_stds)):
        if abs(mean) > 0.001:
            ax1.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width() / 2, mean),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 2. Semantic vs Structural with Error Bars
    ax2 = axes[0, 1]
    
    semantic_types = ['CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING']
    structural_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'BRACKET']
    
    def get_category_aggregate(agg_dict: Dict, categories: List[str]) -> tuple:
        means = [agg_dict.get(cat, {}).get('mean', 0) for cat in categories]
        stds = [agg_dict.get(cat, {}).get('std', 0) for cat in categories]
        total_mean = sum(means)
        # Propagate uncertainty
        total_std = np.sqrt(sum(s**2 for s in stds))
        return total_mean, total_std
    
    natural_sem_mean, natural_sem_std = get_category_aggregate(natural_agg, semantic_types)
    natural_struct_mean, natural_struct_std = get_category_aggregate(natural_agg, structural_types)
    no_cycle_sem_mean, no_cycle_sem_std = get_category_aggregate(no_cycle_agg, semantic_types)
    no_cycle_struct_mean, no_cycle_struct_std = get_category_aggregate(no_cycle_agg, structural_types)
    
    categories = ['Semantic Tokens', 'Structural Tokens']
    natural_means_cat = [natural_sem_mean, natural_struct_mean]
    natural_stds_cat = [natural_sem_std, natural_struct_std]
    no_cycle_means_cat = [no_cycle_sem_mean, no_cycle_struct_mean]
    no_cycle_stds_cat = [no_cycle_sem_std, no_cycle_struct_std]
    
    x_pos = np.arange(len(categories))
    bars3 = ax2.bar(x_pos - width/2, natural_means_cat, width, yerr=natural_stds_cat,
                    label=f'Natural (n={n_seeds} seeds)', alpha=0.8, color='#e74c3c', capsize=5)
    bars4 = ax2.bar(x_pos + width/2, no_cycle_means_cat, width, yerr=no_cycle_stds_cat,
                    label=f'No-Cycle-ICL (n={n_seeds} seeds)', alpha=0.8, color='#3498db', capsize=5)
    
    ax2.set_xlabel('Token Category', fontsize=12)
    ax2.set_ylabel('Total Attention Shift', fontsize=12)
    ax2.set_title(f'🎯 Category Shifts (n={n_seeds} seeds)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar, mean, std in zip(bars3, natural_means_cat, natural_stds_cat):
        ax2.annotate(f'{mean:.3f}±{std:.3f}', xy=(bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar, mean, std in zip(bars4, no_cycle_means_cat, no_cycle_stds_cat):
        ax2.annotate(f'{mean:.3f}±{std:.3f}', xy=(bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Heatmap with Mean Values
    ax3 = axes[1, 0]
    
    shift_matrix = np.array([
        [natural_agg.get(tt, {}).get('mean', 0) for tt in all_token_types],
        [no_cycle_agg.get(tt, {}).get('mean', 0) for tt in all_token_types]
    ])
    
    im = ax3.imshow(shift_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    ax3.set_xticks(range(len(all_token_types)))
    ax3.set_xticklabels(all_token_types, rotation=45, ha='right')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels([f'Natural\n(Repetitive)\nn={n_seeds}', f'No-Cycle-ICL\n(Non-Repetitive)\nn={n_seeds}'])
    ax3.set_title(f'🌡️ Mean Attention Shifts (n={n_seeds} seeds)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(all_token_types)):
            value = shift_matrix[i, j]
            if abs(value) > 0.005:
                ax3.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color='white' if abs(value) > 0.05 else 'black', fontweight='bold')
    
    plt.colorbar(im, ax=ax3, label='Mean Attention Shift')
    
    # 4. Scatter Plot with Confidence Ellipses
    ax4 = axes[1, 1]
    
    key_types = ['CONTENT_WORD', 'FUNCTION_WORD', 'NEWLINE', 'PUNCTUATION', 'PROGRAMMING']
    natural_key_means = [natural_agg.get(kt, {}).get('mean', 0) for kt in key_types]
    natural_key_stds = [natural_agg.get(kt, {}).get('std', 0) for kt in key_types]
    no_cycle_key_means = [no_cycle_agg.get(kt, {}).get('mean', 0) for kt in key_types]
    no_cycle_key_stds = [no_cycle_agg.get(kt, {}).get('std', 0) for kt in key_types]
    
    key_colors = ['#9b59b6', '#e67e22', '#e74c3c', '#f39c12', '#3498db']
    
    # Scatter with error bars
    for i, (tt, color) in enumerate(zip(key_types, key_colors)):
        ax4.errorbar(natural_key_means[i], no_cycle_key_means[i], 
                    xerr=natural_key_stds[i], yerr=no_cycle_key_stds[i],
                    fmt='o', markersize=10, alpha=0.7, color=color, capsize=5, capthick=2)
        ax4.annotate(tt, (natural_key_means[i], no_cycle_key_means[i]), 
                    xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Add diagonal line
    lims = [min(ax4.get_xlim()[0], ax4.get_ylim()[0]), max(ax4.get_xlim()[1], ax4.get_ylim()[1])]
    ax4.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=2)
    
    ax4.set_xlabel('Natural (Repetitive) Mean Shift', fontsize=12)
    ax4.set_ylabel('No-Cycle-ICL (Non-Repetitive) Mean Shift', fontsize=12)
    ax4.set_title(f'📊 Key Token Shifts (n={n_seeds} seeds)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"paper_figure_aggregated_{n_seeds}seeds_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Aggregated plot saved: {plot_path}")
    plt.close()

def create_aggregated_report(natural_agg: Dict, no_cycle_agg: Dict, 
                            all_results: Dict, output_dir: Path, n_seeds: int):
    """Create aggregated analysis report."""
    
    n_natural = len(all_results['natural_results'])
    n_no_cycle = len(all_results['no_cycle_results'])
    
    report = f"""# Aggregated Multi-Seed Attention Fallback Analysis
**OLMo-1B-hf: Natural vs No-Cycle-ICL (n={n_seeds} seeds)**

## Summary

### Data Sources
- **Seeds analyzed**: {', '.join(map(str, all_results['seeds_loaded']))}
- **Total Natural sequences**: {n_natural}
- **Total No-Cycle-ICL sequences**: {n_no_cycle}

### Aggregation Method
Results aggregated across {n_seeds} independent seed runs, each with different random text samples.
Error bars represent standard deviation across seeds.

## Token Type Analysis (Mean ± Std)

| Token Type | Natural Shift | No-Cycle-ICL Shift | Difference | Seeds |
|------------|---------------|-------------------|------------|-------|"""

    all_types = sorted(set(list(natural_agg.keys()) + list(no_cycle_agg.keys())))
    
    for tt in all_types:
        nat = natural_agg.get(tt, {})
        noc = no_cycle_agg.get(tt, {})
        
        nat_mean = nat.get('mean', 0)
        nat_std = nat.get('std', 0)
        noc_mean = noc.get('mean', 0)
        noc_std = noc.get('std', 0)
        diff = nat_mean - noc_mean
        n = nat.get('n_seeds', 0)
        
        report += f"\n| {tt} | {nat_mean:+.3f}±{nat_std:.3f} | {noc_mean:+.3f}±{noc_std:.3f} | {diff:+.3f} | {n} |"
    
    # Category analysis
    semantic_types = ['CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING']
    structural_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'BRACKET']
    
    def get_cat_stats(agg_dict, cats):
        means = [agg_dict.get(c, {}).get('mean', 0) for c in cats]
        stds = [agg_dict.get(c, {}).get('std', 0) for c in cats]
        return sum(means), np.sqrt(sum(s**2 for s in stds))
    
    nat_sem_m, nat_sem_s = get_cat_stats(natural_agg, semantic_types)
    nat_str_m, nat_str_s = get_cat_stats(natural_agg, structural_types)
    noc_sem_m, noc_sem_s = get_cat_stats(no_cycle_agg, semantic_types)
    noc_str_m, noc_str_s = get_cat_stats(no_cycle_agg, structural_types)
    
    report += f"""

## Category-Level Analysis

### Semantic Tokens
- **Natural**: {nat_sem_m:+.3f}±{nat_sem_s:.3f} pp
- **No-Cycle-ICL**: {noc_sem_m:+.3f}±{noc_sem_s:.3f} pp
- **Difference**: {nat_sem_m - noc_sem_m:+.3f} pp

### Structural Tokens
- **Natural**: {nat_str_m:+.3f}±{nat_str_s:.3f} pp
- **No-Cycle-ICL**: {noc_str_m:+.3f}±{noc_str_s:.3f} pp
- **Difference**: {nat_str_m - noc_str_s:+.3f} pp

## Key Findings

### Robustness Across Seeds
Results aggregated from {n_seeds} independent random samples demonstrate:
"""
    
    if n_no_cycle == 0:
        report += "\n⚠️  **No No-Cycle-ICL sequences generated across any seed**\n"
        report += "- OLMo-1B-hf does not produce non-repetitive sequences under these conditions\n"
        report += "- Analysis limited to Natural (repetitive) sequences only\n"
    else:
        report += f"\n✅ **Successful generation across seeds**: {n_no_cycle} No-Cycle-ICL sequences\n"
        
        content_nat = natural_agg.get('CONTENT_WORD', {})
        content_noc = no_cycle_agg.get('CONTENT_WORD', {})
        diff = content_nat.get('mean', 0) - content_noc.get('mean', 0)
        
        if abs(diff) < 0.01:
            report += "- Similar attention fallback patterns between sequence types\n"
        else:
            report += f"- Different fallback strategies: {diff:+.3f} pp difference in content word attention\n"
    
    report += f"""

---
*Aggregated from {n_seeds} seeds with {n_natural + n_no_cycle} total sequences*
*Model: allenai/OLMo-1B-hf*
"""
    
    report_path = output_dir / f"aggregated_{n_seeds}seeds_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Aggregated report saved: {report_path}")

def main():
    print("🔄 Aggregating Multi-Seed Results")
    print("=" * 40)
    
    base_dir = Path("plots")
    output_dir = Path("plots/aggregated_multi_seed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5926, 8192, 16384]
    
    print(f"\n📂 Loading results from {len(seeds)} seeds...")
    all_results = load_all_seed_results(base_dir, seeds)
    
    n_seeds = len(all_results['seeds_loaded'])
    print(f"\n✅ Loaded data from {n_seeds} seeds")
    print(f"   - Total Natural sequences: {len(all_results['natural_results'])}")
    print(f"   - Total No-Cycle-ICL sequences: {len(all_results['no_cycle_results'])}")
    
    if n_seeds == 0:
        print("\n❌ No results found to aggregate")
        return
    
    print("\n📊 Calculating aggregate statistics...")
    natural_agg = calculate_aggregate_shifts(all_results['natural_shifts'])
    no_cycle_agg = calculate_aggregate_shifts(all_results['no_cycle_shifts'])
    
    print(f"\n🎨 Creating aggregated plots...")
    plot_aggregated_comparison(natural_agg, no_cycle_agg, output_dir, n_seeds)
    
    print(f"\n📝 Creating aggregated report...")
    create_aggregated_report(natural_agg, no_cycle_agg, all_results, output_dir, n_seeds)
    
    print(f"\n✅ Aggregation complete!")
    print(f"📁 Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
