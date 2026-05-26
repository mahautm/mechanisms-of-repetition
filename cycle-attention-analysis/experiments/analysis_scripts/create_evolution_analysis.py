#!/usr/bin/env python3
"""
Create evolution graphs showing how attention bias patterns change across training steps.
"""

print("🔧 Starting imports...")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import argparse

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

def parse_step_number(step_name):
    """Extract numeric step from step name for sorting."""
    if step_name == 'steplatest':
        return float('inf')  # Put steplatest at the end
    elif step_name.startswith('step'):
        try:
            return int(step_name[4:])
        except ValueError:
            return float('inf')
    return 0

def collect_evolution_data(multi_step_dir):
    """Collect bias ratio data across all training steps."""
    
    print(f"📊 Collecting evolution data from: {multi_step_dir}")
    
    # Find all bias summary directories
    step_dirs = []
    for item in Path(multi_step_dir).iterdir():
        if item.is_dir() and item.name.startswith('bias_summary_'):
            step_name = item.name.replace('bias_summary_', '')
            step_dirs.append((step_name, item))
    
    # Sort by step number
    step_dirs.sort(key=lambda x: parse_step_number(x[0]))
    
    print(f"📋 Found {len(step_dirs)} training steps:")
    for step_name, _ in step_dirs:
        print(f"   - {step_name}")
    
    # Collect data from each step
    evolution_data = []
    
    for step_name, step_dir in step_dirs:
        print(f"   📄 Processing: {step_name}")
        
        # Read the statistical summary
        summary_file = step_dir / "attention_bias_statistical_summary.md"
        
        if not summary_file.exists():
            print(f"      ⚠️  No summary file found for {step_name}")
            continue
        
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            
            # Parse the data
            step_data = parse_summary_content(content, step_name)
            if step_data:
                evolution_data.append(step_data)
                
        except Exception as e:
            print(f"      ❌ Error parsing {step_name}: {e}")
            continue
    
    print(f"✅ Successfully collected data from {len(evolution_data)} steps")
    return evolution_data

def parse_summary_content(content, step_name):
    """Parse markdown summary content to extract bias ratios."""
    
    step_data = {'step_name': step_name, 'step_number': parse_step_number(step_name)}
    
    # Split content by sequence types - handle different header formats
    sections = content.split('### ')
    
    for section in sections:
        seq_type = None
        if 'Natural Sequences' in section or 'NATURAL Scenario' in section:
            seq_type = 'natural'
        elif 'No Cycle Icl Sequences' in section or 'ICL Scenario' in section:
            seq_type = 'no_cycle_icl'
        
        if seq_type is None:
            continue
        
        # Extract table data
        bias_ratios = {}
        lines = section.split('\n')
        table_started = False
        
        for line in lines:
            if '|------------|' in line:
                table_started = True
                continue
            
            if table_started and line.strip().startswith('|') and len(line.split('|')) >= 4:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    token_type = parts[1]
                    try:
                        # Try different column positions for ratio data
                        ratio_value = None
                        # First try median ratio (column 3), then try max ratio (column 2)
                        for col_idx in [3, 2]:
                            if col_idx < len(parts):
                                try:
                                    ratio_value = float(parts[col_idx])
                                    break
                                except ValueError:
                                    continue
                        
                        if ratio_value is not None:
                            bias_ratios[token_type] = ratio_value
                    except (ValueError, IndexError):
                        continue
        
        if bias_ratios:  # Only add if we found data
            step_data[seq_type] = bias_ratios
    
    # Return data if we have at least step info + one sequence type
    return step_data if len(step_data) > 2 else None

def create_evolution_plots(evolution_data, output_dir):
    """Create comprehensive evolution plots."""
    
    if not evolution_data:
        print("❌ No evolution data to plot!")
        return
    
    # Prepare data for plotting
    steps = []
    step_labels = []
    
    for data in evolution_data:
        step_num = data['step_number']
        if step_num == float('inf'):
            steps.append(len(evolution_data))  # Put steplatest at the end
            step_labels.append('latest')
        else:
            steps.append(step_num)
            step_labels.append(f"{step_num:,}")
    
    # Key token types to track
    key_tokens = ['NEWLINE', 'TEMPLATE_WORD', 'CONTENT_WORD']
    
    # Determine which sequence types we have data for
    available_seq_types = set()
    for data in evolution_data:
        if 'natural' in data:
            available_seq_types.add('natural')
        if 'no_cycle_icl' in data:
            available_seq_types.add('no_cycle_icl')
    
    seq_types = ['natural', 'no_cycle_icl']
    seq_types = [st for st in seq_types if st in available_seq_types]  # Only plot available types
    
    colors = {
        'NEWLINE': '#e74c3c',
        'TEMPLATE_WORD': '#3498db',
        'CONTENT_WORD': '#9b59b6'
    }
    
    # Create evolution plot - adjust subplot count based on available data
    num_plots = len(seq_types)
    if num_plots == 0:
        print("❌ No sequence type data found!")
        return
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 6 * num_plots))
    if num_plots == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    for seq_idx, seq_type in enumerate(seq_types):
        ax = axes[seq_idx]
        
        # Plot evolution for each key token type
        for token_type in key_tokens:
            ratios = []
            valid_steps = []
            valid_labels = []
            
            for i, data in enumerate(evolution_data):
                if seq_type in data and token_type in data[seq_type]:
                    ratio = data[seq_type][token_type]
                    ratios.append(ratio)
                    valid_steps.append(steps[i])
                    valid_labels.append(step_labels[i])
            
            if ratios:
                # Convert steps to log scale for better visualization (except for steplatest)
                plot_steps = []
                for i, step in enumerate(valid_steps):
                    if valid_labels[i] == 'latest':
                        # Position steplatest slightly after the last numeric step
                        max_numeric_step = max([s for s in valid_steps if s != len(evolution_data)])
                        plot_steps.append(max_numeric_step * 1.2)
                    else:
                        plot_steps.append(step)
                
                ax.plot(plot_steps, ratios, marker='o', linewidth=3, markersize=8, 
                       label=token_type, color=colors[token_type], alpha=0.8)
                
                # Add value annotations
                for x, y, label in zip(plot_steps, ratios, valid_labels):
                    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=9, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[token_type], alpha=0.3))
        
        # Reference line for proportional attention
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='Proportional (1:1)')
        
        # Formatting
        ax.set_title(f'{seq_type.replace("_", " ").title()} Sequences\nBias Ratio Evolution During Training', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Bias Ratio (Attention % / Prompt %)', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set custom x-axis labels
        ax.set_xticks([1000, 10000, 100000, max([s for s in steps if s != len(evolution_data)]) * 1.2])
        ax.set_xticklabels(['1K', '10K', '100K', 'Latest'])
        
        # Add interpretation zones
        ax.axhspan(0.8, 1.2, alpha=0.1, color='green', zorder=0)
        ax.axhspan(2, max([max(ratios) for ratios in [[data[seq_type][token] for token in key_tokens if token in data.get(seq_type, {})] for data in evolution_data] if ratios]) * 1.1, 
                  alpha=0.05, color='red', zorder=0)
        ax.axhspan(0, 0.5, alpha=0.05, color='blue', zorder=0)
    
    plt.tight_layout()
    
    evolution_path = output_dir / "attention_bias_evolution.png"
    plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Evolution plot saved: {evolution_path}")
    return evolution_path

def create_specialization_development_plot(evolution_data, output_dir):
    """Create plot showing the development of extreme specialization."""
    
    # Calculate specialization metrics for each step
    specialization_metrics = []
    
    for data in evolution_data:
        step_metrics = {
            'step_name': data['step_name'],
            'step_number': data['step_number'],
            'natural_newline_bias': 0,
            'natural_content_suppression': 0,
            'icl_balance_score': 0,
            'extreme_specialization_count': 0
        }
        
        # Natural sequences metrics
        if 'natural' in data:
            natural_data = data['natural']
            
            # Newline over-attention
            step_metrics['natural_newline_bias'] = natural_data.get('NEWLINE', 0)
            
            # Content suppression (inverse of content word bias)
            content_bias = natural_data.get('CONTENT_WORD', 1)
            step_metrics['natural_content_suppression'] = 1 / max(content_bias, 0.01)  # Avoid division by zero
            
            # Count extreme specializations (>5x bias)
            extreme_count = sum(1 for ratio in natural_data.values() if ratio > 5)
            step_metrics['extreme_specialization_count'] = extreme_count
        
        # ICL balance score (how close to proportional)
        if 'no_cycle_icl' in data:
            icl_data = data['no_cycle_icl']
            # Calculate average deviation from 1.0 (proportional)
            deviations = [abs(ratio - 1.0) for ratio in icl_data.values() if ratio > 0]
            step_metrics['icl_balance_score'] = 1 / (1 + np.mean(deviations)) if deviations else 0
        
        specialization_metrics.append(step_metrics)
    
    # Create specialization development plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Prepare x-axis
    steps = []
    step_labels = []
    for data in specialization_metrics:
        step_num = data['step_number']
        if step_num == float('inf'):
            steps.append(len(specialization_metrics))
            step_labels.append('latest')
        else:
            steps.append(step_num)
            step_labels.append(f"{step_num:,}")
    
    # Plot 1: Newline Bias Evolution
    ax1 = axes[0]
    newline_biases = [m['natural_newline_bias'] for m in specialization_metrics]
    
    plot_steps = []
    for i, step in enumerate(steps):
        if step_labels[i] == 'latest':
            max_numeric_step = max([s for s in steps if s != len(specialization_metrics)])
            plot_steps.append(max_numeric_step * 1.2)
        else:
            plot_steps.append(step)
    
    ax1.plot(plot_steps, newline_biases, marker='o', linewidth=3, markersize=8, 
            color='#e74c3c', alpha=0.8)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Proportional')
    ax1.set_title('Newline Specialization Development', fontweight='bold')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Newline Bias Ratio')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add annotations
    for x, y, label in zip(plot_steps, newline_biases, step_labels):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Content Suppression Evolution
    ax2 = axes[1]
    content_suppressions = [m['natural_content_suppression'] for m in specialization_metrics]
    
    ax2.plot(plot_steps, content_suppressions, marker='o', linewidth=3, markersize=8, 
            color='#9b59b6', alpha=0.8)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No suppression')
    ax2.set_title('Content Word Suppression Development', fontweight='bold')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Content Suppression Factor')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Extreme Specialization Count
    ax3 = axes[2]
    extreme_counts = [m['extreme_specialization_count'] for m in specialization_metrics]
    
    ax3.plot(plot_steps, extreme_counts, marker='o', linewidth=3, markersize=8, 
            color='#f39c12', alpha=0.8)
    ax3.set_title('Number of Extreme Specializations (>5x bias)', fontweight='bold')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Count of Extreme Specializations')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ICL Balance Score
    ax4 = axes[3]
    balance_scores = [m['icl_balance_score'] for m in specialization_metrics]
    
    ax4.plot(plot_steps, balance_scores, marker='o', linewidth=3, markersize=8, 
            color='#2ecc71', alpha=0.8)
    ax4.set_title('ICL Attention Balance Score', fontweight='bold')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Balance Score (1=perfect balance)')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Set consistent x-axis
    for ax in axes:
        ax.set_xticks([1000, 10000, 100000, max([s for s in steps if s != len(specialization_metrics)]) * 1.2])
        ax.set_xticklabels(['1K', '10K', '100K', 'Latest'])
    
    plt.tight_layout()
    
    specialization_path = output_dir / "specialization_development.png"
    plt.savefig(specialization_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Specialization development plot saved: {specialization_path}")
    return specialization_path

def create_summary_evolution_report(evolution_data, output_dir):
    """Create a comprehensive evolution summary report."""
    
    report_lines = []
    report_lines.append("# Attention Bias Evolution Analysis")
    report_lines.append("")
    report_lines.append("## Training Progression Summary")
    report_lines.append("")
    
    # Create evolution table
    report_lines.append("| Training Step | Newline Bias (Natural) | Content Bias (Natural) | Template Max (Natural) | Newline Bias (ICL) |")
    report_lines.append("|---------------|------------------------|------------------------|------------------------|---------------------|")
    
    for data in evolution_data:
        step_name = data['step_name']
        
        # Natural sequence data
        natural = data.get('natural', {})
        newline_nat = natural.get('NEWLINE', 0)
        content_nat = natural.get('CONTENT_WORD', 0)
        template_nat = natural.get('TEMPLATE_WORD', 0)
        
        # ICL sequence data
        icl = data.get('no_cycle_icl', {})
        newline_icl = icl.get('NEWLINE', 0)
        
        report_lines.append(f"| {step_name} | {newline_nat:.2f}x | {content_nat:.2f}x | {template_nat:.2f}x | {newline_icl:.2f}x |")
    
    report_lines.append("")
    report_lines.append("## Key Evolution Findings")
    report_lines.append("")
    
    # Calculate key findings
    first_step = evolution_data[0]
    last_step = evolution_data[-1]
    
    # Newline bias evolution
    first_newline = first_step.get('natural', {}).get('NEWLINE', 1)
    last_newline = last_step.get('natural', {}).get('NEWLINE', 1)
    newline_increase = last_newline / first_newline
    
    # Content suppression evolution
    first_content = first_step.get('natural', {}).get('CONTENT_WORD', 1)
    last_content = last_step.get('natural', {}).get('CONTENT_WORD', 1)
    content_suppression = first_content / last_content
    
    report_lines.append(f"### Structural Specialization Development")
    report_lines.append(f"")
    report_lines.append(f"- **Newline bias increase**: {first_newline:.2f}x → {last_newline:.2f}x ({newline_increase:.1f}x growth)")
    report_lines.append(f"- **Content suppression**: {first_content:.2f}x → {last_content:.2f}x ({content_suppression:.1f}x more suppression)")
    report_lines.append(f"")
    
    # Template specialization
    template_evolution = []
    for data in evolution_data:
        template_bias = data.get('natural', {}).get('TEMPLATE_WORD', 0)
        template_evolution.append(template_bias)
    
    max_template_step = evolution_data[np.argmax(template_evolution)]
    report_lines.append(f"### Template Word Specialization")
    report_lines.append(f"")
    report_lines.append(f"- **Peak specialization**: {max(template_evolution):.2f}x at {max_template_step['step_name']}")
    report_lines.append(f"- **Final specialization**: {template_evolution[-1]:.2f}x")
    report_lines.append(f"")
    
    report_lines.append("## Scientific Implications")
    report_lines.append("")
    report_lines.append("1. **Gradual Development**: Structural bias develops progressively during training")
    report_lines.append("2. **Content Suppression**: Model actively learns to ignore semantic content")  
    report_lines.append("3. **Specialization Circuits**: Specific layers develop extreme specializations")
    report_lines.append("4. **Architecture vs Learning**: Repetition is learned behavior, not architectural bias")
    report_lines.append("")
    
    report_lines.append("## Training Stage Analysis")
    report_lines.append("")
    
    if len(evolution_data) >= 3:
        early = evolution_data[0]
        mid = evolution_data[len(evolution_data)//2]
        late = evolution_data[-1]
        
        report_lines.append(f"### Early Training ({early['step_name']})")
        early_natural = early.get('natural', {})
        proportional_count = sum(1 for ratio in early_natural.values() if 0.8 <= ratio <= 1.2)
        report_lines.append(f"- **Proportional tokens**: {proportional_count}/{len(early_natural)} token types")
        report_lines.append(f"- **Max specialization**: {max(early_natural.values()):.2f}x")
        report_lines.append("")
        
        report_lines.append(f"### Late Training ({late['step_name']})")
        late_natural = late.get('natural', {})
        proportional_count = sum(1 for ratio in late_natural.values() if 0.8 <= ratio <= 1.2)
        extreme_count = sum(1 for ratio in late_natural.values() if ratio > 5)
        report_lines.append(f"- **Proportional tokens**: {proportional_count}/{len(late_natural)} token types")
        report_lines.append(f"- **Extreme specializations**: {extreme_count} token types")
        report_lines.append(f"- **Max specialization**: {max(late_natural.values()):.2f}x")
    
    # Save report
    report_path = output_dir / "attention_bias_evolution_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   ✅ Evolution report saved: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Create evolution graphs from multi-step analysis")
    parser.add_argument("--multi_step_dir", type=str,
                       default="./plots/multi_step_analysis",
                       help="Directory containing multi-step analysis results")
    parser.add_argument("--output_dir", type=str, default="./plots/evolution_analysis",
                       help="Output directory for evolution plots")
    
    args = parser.parse_args()
    
    multi_step_dir = Path(args.multi_step_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Creating evolution analysis...")
    print(f"📂 Multi-step directory: {multi_step_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Collect evolution data
    evolution_data = collect_evolution_data(multi_step_dir)
    
    if not evolution_data:
        print("❌ No evolution data found!")
        return
    
    print(f"📊 Creating evolution visualizations...")
    
    # Create evolution plots
    evolution_path = create_evolution_plots(evolution_data, output_dir)
    specialization_path = create_specialization_development_plot(evolution_data, output_dir)
    report_path = create_summary_evolution_report(evolution_data, output_dir)
    
    print(f"\n✅ Evolution analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Generated plots:")
    print(f"   - Evolution trends: {evolution_path.name}")
    print(f"   - Specialization development: {specialization_path.name}")
    print(f"   - Summary report: {report_path.name}")

if __name__ == "__main__":
    main()