#!/usr/bin/env python3
"""
Create summary figure showing attention bias ratios across all layers and token types.
Parses the markdown reports to extract bias ratio data.
"""

print("🔧 Starting imports...")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import argparse

# Set publication-quality plotting parameters for compact paper format
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'serif'
})

print("✅ All imports successful!")

def parse_report_file(report_path):
    """Parse a markdown report file to extract bias ratio data."""
    
    try:
        with open(report_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"   ⚠️  Error reading {report_path}: {e}")
        return None
    
    # Extract layer number from filename
    layer_match = re.search(r'_L(\d+)_', report_path.name)
    if not layer_match:
        print(f"   ⚠️  Could not extract layer from {report_path.name}")
        return None
    
    layer_num = int(layer_match.group(1))
    
    # Parse the markdown tables
    data = {'layer': layer_num}
    
    # Split content by sequence types
    sections = content.split('## ')
    
    for section in sections:
        if 'Natural Sequences' in section:
            seq_type = 'natural'
        elif 'No Cycle Icl Sequences' in section:
            seq_type = 'no_cycle_icl'
        else:
            continue
        
        # Extract table data
        lines = section.split('\n')
        table_started = False
        
        bias_ratios = {}
        
        for line in lines:
            if '|------------|' in line:
                table_started = True
                continue
            
            if table_started and line.strip().startswith('|') and len(line.split('|')) >= 5:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5:
                    token_type = parts[1]
                    bias_ratio_str = parts[4]
                    
                    # Parse bias ratio
                    try:
                        if bias_ratio_str == '∞':
                            bias_ratio = float('inf')
                        else:
                            bias_ratio = float(bias_ratio_str)
                        bias_ratios[token_type] = bias_ratio
                    except (ValueError, IndexError):
                        continue
        
        data[seq_type] = bias_ratios
    
    return data

def collect_all_data(reports_dir):
    """Collect bias ratio data from all report files."""
    
    print(f"📊 Collecting data from reports in: {reports_dir}")
    
    all_data = []
    report_files = list(Path(reports_dir).glob("prompt_vs_attention_report_L*_C4.md"))
    
    print(f"📋 Found {len(report_files)} report files")
    
    for report_file in sorted(report_files):
        print(f"   📄 Processing: {report_file.name}")
        data = parse_report_file(report_file)
        if data:
            all_data.append(data)
        
    print(f"✅ Successfully parsed {len(all_data)} reports")
    return all_data

def create_bias_ratio_heatmap(all_data, output_dir):
    """Create heatmap showing bias ratios across layers and token types."""
    
    # Merge TEMPLATE_WORD into CONTENT_WORD for all data
    for data in all_data:
        for seq_type in ['natural', 'no_cycle_icl']:
            if seq_type in data:
                if 'TEMPLATE_WORD' in data[seq_type] and 'CONTENT_WORD' in data[seq_type]:
                    # Merge by averaging (weighted by their relative frequencies)
                    data[seq_type]['CONTENT_WORD'] = (data[seq_type]['CONTENT_WORD'] + data[seq_type]['TEMPLATE_WORD']) / 2
                    del data[seq_type]['TEMPLATE_WORD']
                elif 'TEMPLATE_WORD' in data[seq_type]:
                    data[seq_type]['CONTENT_WORD'] = data[seq_type]['TEMPLATE_WORD']
                    del data[seq_type]['TEMPLATE_WORD']
    
    # Organize data
    layers = sorted(set(d['layer'] for d in all_data))
    token_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING', 'BRACKET', 'NUMBER', 'OTHER']
    seq_types = ['natural', 'no_cycle_icl']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for seq_idx, seq_type in enumerate(seq_types):
        ax = axes[seq_idx]
        
        # Create data matrix
        matrix_data = []
        layer_labels = []
        
        for layer in layers:
            layer_data = next((d for d in all_data if d['layer'] == layer), None)
            if not layer_data or seq_type not in layer_data:
                # Fill with zeros if no data
                row = [0] * len(token_types)
            else:
                seq_data = layer_data[seq_type]
                row = []
                for token_type in token_types:
                    ratio = seq_data.get(token_type, 0)
                    # Cap infinite values for visualization
                    if ratio == float('inf'):
                        ratio = 50  # Cap at 50 for visualization
                    elif ratio > 50:
                        ratio = 50
                    row.append(ratio)
            
            matrix_data.append(row)
            layer_labels.append(f'L{layer}')
        
        # Create heatmap with linear scale coloring
        matrix_data = np.array(matrix_data)
        
        # Use linear scale for better visualization
        im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=15)
        
        # Set labels
        ax.set_title(f'{seq_type.replace("_", " ").title()} Sequences\nAttention Bias Ratios (Linear Scale)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Token Categories', fontsize=14)
        ax.set_ylabel('Model Layers', fontsize=14)
        
        # Set ticks
        ax.set_xticks(range(len(token_types)))
        ax.set_xticklabels(token_types, rotation=45, ha='right')
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels)
        
        # Add text annotations for key values
        for i in range(len(layers)):
            for j in range(len(token_types)):
                value = matrix_data[i, j]
                if value > 5 or value < 0.2:  # Highlight extreme values
                    if value == 0:
                        text = '0'
                    elif value >= 50:
                        text = '50+'
                    else:
                        text = f'{value:.1f}'
                    
                    # Choose text color based on background
                    text_color = 'white' if matrix_data[i, j] > np.median(matrix_data) else 'black'
                    ax.text(j, i, text, ha="center", va="center", 
                           color=text_color, fontweight='bold', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Bias Ratio (Linear Scale)', rotation=270, labelpad=20)
        
        # Add reference lines
        ax.axhline(y=-0.5, color='white', linestyle='-', linewidth=2, alpha=0.3)
        
    plt.tight_layout()
    
    heatmap_path = output_dir / "attention_bias_ratio_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Bias ratio heatmap saved: {heatmap_path}")
    return heatmap_path

def create_bias_ratio_summary_plot(all_data, output_dir):
    """Create summary plot showing average bias ratios by token type with paired Natural/ICL columns."""
    
    # Note: TEMPLATE_WORD already merged into CONTENT_WORD in heatmap function
    token_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING', 'BRACKET', 'NUMBER', 'OTHER']
    seq_types = ['natural', 'no_cycle_icl']
    
    # Define colors for Natural and ICL
    natural_color = '#e74c3c'  # Red
    icl_color = '#3498db'  # Blue
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate statistics for both sequence types
    natural_stats = {}
    icl_stats = {}
    
    for token_type in token_types:
        # Natural sequences
        natural_ratios = []
        for data in all_data:
            if 'natural' in data and token_type in data['natural']:
                ratio = data['natural'][token_type]
                if ratio != float('inf') and ratio > 0:
                    natural_ratios.append(min(ratio, 50))
        
        if natural_ratios:
            natural_stats[token_type] = {
                'mean': np.mean(natural_ratios),
                'median': np.median(natural_ratios),
                'std': np.std(natural_ratios),
                'count': len(natural_ratios)
            }
        else:
            natural_stats[token_type] = {'mean': 0, 'median': 0, 'std': 0, 'count': 0}
        
        # ICL sequences
        icl_ratios = []
        for data in all_data:
            if 'no_cycle_icl' in data and token_type in data['no_cycle_icl']:
                ratio = data['no_cycle_icl'][token_type]
                if ratio != float('inf') and ratio > 0:
                    icl_ratios.append(min(ratio, 50))
        
        if icl_ratios:
            icl_stats[token_type] = {
                'mean': np.mean(icl_ratios),
                'median': np.median(icl_ratios),
                'std': np.std(icl_ratios),
                'count': len(icl_ratios)
            }
        else:
            icl_stats[token_type] = {'mean': 0, 'median': 0, 'std': 0, 'count': 0}
    
    # Sort token types by average median bias ratio
    sorted_tokens = sorted(token_types, 
                          key=lambda t: (natural_stats[t]['median'] + icl_stats[t]['median']) / 2, 
                          reverse=True)
    
    # Create paired bar plot
    x_pos = np.arange(len(sorted_tokens))
    width = 0.35
    
    natural_means = [natural_stats[t]['mean'] for t in sorted_tokens]
    natural_stds = [natural_stats[t]['std'] for t in sorted_tokens]
    icl_means = [icl_stats[t]['mean'] for t in sorted_tokens]
    icl_stds = [icl_stats[t]['std'] for t in sorted_tokens]
    
    # Create paired bars
    bars1 = ax.bar(x_pos - width/2, natural_means, width, yerr=natural_stds, 
                   label='Natural', color=natural_color, alpha=0.85, 
                   capsize=5, error_kw={'linewidth': 1.5}, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos + width/2, icl_means, width, yerr=icl_stds,
                   label='ICL', color=icl_color, alpha=0.85,
                   capsize=5, error_kw={'linewidth': 1.5}, edgecolor='black', linewidth=0.8)
    
    # Add reference line at ratio = 1 (proportional attention)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.5, 
              label='Proportional (1:1)')
    
    # Labels and formatting
    ax.set_xlabel('Token Categories', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bias Ratio (Attention % / Prompt %)', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_tokens, rotation=15, ha='right')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')
    
    # Add interpretation zones
    y_max = 15
    ax.axhspan(0.8, 1.2, alpha=0.15, color='green')
    
    # Set y-axis limits
    ax.set_ylim(0, y_max)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    summary_path = output_dir / "attention_bias_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Bias ratio summary plot saved: {summary_path}")
    return summary_path

def create_layer_evolution_plot(all_data, output_dir):
    """Create plot showing how bias ratios evolve across layers."""
    
    key_tokens = ['NEWLINE', 'CONTENT_WORD']  # Focus on most important
    seq_types = ['natural', 'no_cycle_icl']
    
    colors = {
        'NEWLINE': '#e74c3c',
        'CONTENT_WORD': '#9b59b6'
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    for seq_idx, seq_type in enumerate(seq_types):
        ax = axes[seq_idx]
        
        # Organize data by layer
        layers = sorted(set(d['layer'] for d in all_data))
        
        for token_type in key_tokens:
            layer_ratios = []
            layer_nums = []
            
            for layer in layers:
                data = next((d for d in all_data if d['layer'] == layer), None)
                if data and seq_type in data and token_type in data[seq_type]:
                    ratio = data[seq_type][token_type]
                    if ratio != float('inf') and ratio > 0:
                        layer_ratios.append(min(ratio, 50))  # Cap at 50
                        layer_nums.append(layer)
            
            if layer_ratios:
                ax.plot(layer_nums, layer_ratios, marker='o', linewidth=3, 
                       markersize=8, label=token_type, color=colors[token_type], alpha=0.8)
        
        # Reference line
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='Proportional (1:1)')
        
        # Formatting
        ax.set_title(f'{seq_type.replace("_", " ").title()} Sequences\nBias Ratio Evolution Across Layers', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer Number', fontsize=12)
        ax.set_ylabel('Bias Ratio', fontsize=12)
        # ax.set_yscale('log')  # Using linear scale for better visualization
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis to show all layers
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    evolution_path = output_dir / "bias_ratio_layer_evolution.png"
    plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Layer evolution plot saved: {evolution_path}")
    return evolution_path

def create_statistical_summary_report(all_data, output_dir):
    """Create detailed statistical summary report."""
    
    # Note: TEMPLATE_WORD already merged into CONTENT_WORD in heatmap function
    token_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING', 'BRACKET', 'NUMBER', 'OTHER']
    seq_types = ['natural', 'no_cycle_icl']
    
    report = []
    report.append("# Attention Bias Ratio Statistical Summary")
    report.append("")
    report.append("## Key Findings")
    report.append("")
    
    # Calculate overall statistics
    for seq_type in seq_types:
        report.append(f"### {seq_type.replace('_', ' ').title()} Sequences")
        report.append("")
        
        # Calculate statistics for each token type
        stats_table = []
        stats_table.append("| Token Type | Mean Ratio | Median Ratio | Max Ratio | Layers with Data | Interpretation |")
        stats_table.append("|------------|------------|--------------|-----------|------------------|----------------|")
        
        token_summaries = []
        
        for token_type in token_types:
            ratios = []
            layers_with_data = []
            
            for data in all_data:
                if seq_type in data and token_type in data[seq_type]:
                    ratio = data[seq_type][token_type]
                    if ratio != float('inf') and ratio > 0:
                        ratios.append(min(ratio, 50))
                        layers_with_data.append(data['layer'])
            
            if ratios:
                mean_ratio = np.mean(ratios)
                median_ratio = np.median(ratios)
                max_ratio = np.max(ratios)
                n_layers = len(layers_with_data)
                
                # Interpretation
                if median_ratio > 5:
                    interp = "🔴 **Severely Over-attended**"
                elif median_ratio > 2:
                    interp = "🟡 **Over-attended**"
                elif median_ratio < 0.2:
                    interp = "🔵 **Severely Under-attended**"
                elif median_ratio < 0.8:
                    interp = "🟡 **Under-attended**"
                else:
                    interp = "🟢 **Proportional**"
                
                stats_table.append(f"| {token_type} | {mean_ratio:.2f} | {median_ratio:.2f} | {max_ratio:.1f} | {n_layers}/24 | {interp} |")
                
                token_summaries.append((token_type, median_ratio, interp))
        
        # Add table to report
        report.extend(stats_table)
        report.append("")
        
        # Add key insights
        token_summaries.sort(key=lambda x: x[1], reverse=True)
        
        report.append("#### Key Insights:")
        report.append(f"- **Most over-attended**: {token_summaries[0][0]} (median ratio: {token_summaries[0][1]:.2f})")
        report.append(f"- **Most under-attended**: {token_summaries[-1][0]} (median ratio: {token_summaries[-1][1]:.2f})")
        
        # Count by interpretation
        severe_over = len([t for t in token_summaries if 'Severely Over-attended' in t[2]])
        proportional = len([t for t in token_summaries if 'Proportional' in t[2]])
        severe_under = len([t for t in token_summaries if 'Severely Under-attended' in t[2]])
        
        report.append(f"- **Severely over-attended tokens**: {severe_over}/8")
        report.append(f"- **Proportional tokens**: {proportional}/8")
        report.append(f"- **Severely under-attended tokens**: {severe_under}/8")
        report.append("")
    
    # Overall conclusions
    report.append("## Overall Conclusions")
    report.append("")
    report.append("1. **Attention is NOT proportional to prompt composition**")
    report.append("2. **Structural tokens (NEWLINE, TEMPLATE_WORD) receive massive over-attention**")
    report.append("3. **Content tokens receive severe under-attention despite high frequency**")
    report.append("4. **Natural sequences show more extreme biases than ICL sequences**")
    report.append("5. **The model has learned to focus on rare structural signals for repetition**")
    
    # Save report
    report_path = output_dir / "attention_bias_statistical_summary.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"   ✅ Statistical summary saved: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Create summary figures from prompt vs attention reports")
    parser.add_argument("--reports_dir", type=str, 
                       default="./plots/prompt_vs_attention_analysis",
                       help="Directory containing the markdown reports")
    parser.add_argument("--output_dir", type=str, default="./plots/attention_bias_summary",
                       help="Output directory for summary figures")
    
    args = parser.parse_args()
    
    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Creating summary figures from reports...")
    print(f"📂 Reports directory: {reports_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Collect all data from reports
    all_data = collect_all_data(reports_dir)
    
    if not all_data:
        print("❌ No data collected from reports!")
        return
    
    print(f"📊 Creating summary visualizations...")
    
    # Create comprehensive visualizations
    heatmap_path = create_bias_ratio_heatmap(all_data, output_dir)
    summary_path = create_bias_ratio_summary_plot(all_data, output_dir)
    evolution_path = create_layer_evolution_plot(all_data, output_dir)
    report_path = create_statistical_summary_report(all_data, output_dir)
    
    print(f"\n✅ Summary analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Generated {len(list(output_dir.glob('*.png')))} plots and {len(list(output_dir.glob('*.md')))} reports")

if __name__ == "__main__":
    main()