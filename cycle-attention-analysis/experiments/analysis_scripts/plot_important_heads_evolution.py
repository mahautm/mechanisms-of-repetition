#!/usr/bin/env python3
"""
Create focused cycle evolution plots for heads identified as important for repetitions.
A head is considered important if it's in the top 10 highest contrast values at cycle 3.
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

print("✅ All imports successful!")

def extract_contrast_values(base_path, cycle_size=3, t_size=32):
    """
    Extract contrast values from log files for each layer and head.
    Returns a dictionary with layer -> head -> contrast_value mapping.
    """
    print(f"📊 Extracting contrast values from: {base_path}")
    print(f"   - cycle_size: {cycle_size}")
    print(f"   - t_size: {t_size}")
    
    contrast_data = {}
    
    # Process each layer directory
    for layer_dir in sorted(Path(base_path).glob("layer_*")):
        layer_num = int(layer_dir.name.split('_')[1])
        print(f"   Processing layer {layer_num}...")
        
        # Look for the appropriate log file
        log_files = list(layer_dir.glob(f"*_cyc{cycle_size}_*{t_size}.out"))
        if not log_files:
            print(f"     ⚠️ No log file found for layer {layer_num}")
            continue
            
        log_file = log_files[0]  # Take first matching file
        
        # Read and parse the log file
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Extract natural and no-cycle icl heatmaps
            natural_match = re.search(fr"layer {layer_num} natural heatmap: (.+)", log_content)
            no_cycle_icl_match = re.search(fr"layer {layer_num} no-cycle icl heatmap: (.+)", log_content)
            
            if natural_match:
                natural_values = np.array(eval(natural_match.group(1)))
                contrast_data.setdefault(layer_num, {})['natural'] = natural_values
                
            if no_cycle_icl_match:
                no_cycle_icl_values = np.array(eval(no_cycle_icl_match.group(1)))
                contrast_data.setdefault(layer_num, {})['no_cycle_icl'] = no_cycle_icl_values
                
            print(f"     ✅ Extracted data for layer {layer_num}")
            
        except Exception as e:
            print(f"     ❌ Error processing layer {layer_num}: {e}")
            continue
    
    return contrast_data

def identify_important_heads(contrast_data, top_k=10):
    """
    Identify top K heads with highest contrast values across all layers.
    Returns list of (layer, head) tuples.
    """
    print(f"🎯 Identifying top {top_k} important heads...")
    
    all_contrasts = []
    
    # Collect all contrast values with their positions
    for layer_num, layer_data in contrast_data.items():
        for seq_type, values in layer_data.items():
            if seq_type in ['natural', 'no_cycle_icl']:  # Only consider these types
                for head_idx, contrast in enumerate(values):
                    # Use absolute value for ranking importance
                    all_contrasts.append((abs(contrast), layer_num, head_idx, seq_type))
    
    # Sort by contrast value (descending)
    all_contrasts.sort(reverse=True)
    
    # Get top K unique (layer, head) pairs
    important_heads = set()
    for contrast_val, layer_num, head_idx, seq_type in all_contrasts:
        if len(important_heads) >= top_k:
            break
        important_heads.add((layer_num, head_idx))
    
    important_heads = sorted(list(important_heads))
    
    print(f"   ✅ Top {len(important_heads)} important heads:")
    for layer, head in important_heads:
        contrast_val = max(
            abs(contrast_data.get(layer, {}).get('natural', [0])[head] if head < len(contrast_data.get(layer, {}).get('natural', [])) else 0),
            abs(contrast_data.get(layer, {}).get('no_cycle_icl', [0])[head] if head < len(contrast_data.get(layer, {}).get('no_cycle_icl', [])) else 0)
        )
        print(f"     Layer {layer}, Head {head}: contrast = {contrast_val:.2e}")
    
    return important_heads

def load_cycle_evolution_data(data_path):
    """
    Load existing cycle evolution data from .pt files.
    """
    print(f"📥 Loading cycle evolution data from: {data_path}")
    
    all_data = {}
    
    for pt_file in Path(data_path).glob("cycle_evolution_parametric_c4_l*_all_results.pt"):
        try:
            # Extract layer number from filename
            layer_match = re.search(r'_l(\d+)_', pt_file.name)
            if not layer_match:
                continue
            layer_num = int(layer_match.group(1))
            
            # Load the data
            data = torch.load(pt_file, map_location='cpu')
            all_data[layer_num] = data
            
            print(f"   ✅ Loaded layer {layer_num} data")
            
        except Exception as e:
            print(f"   ❌ Error loading {pt_file}: {e}")
    
    return all_data

def plot_focused_evolution(cycle_data, important_heads, seq_types=['natural', 'no_cycle_icl'], 
                          plot_type='focus', output_dir=None):
    """
    Create focused plots showing only the important heads.
    """
    if output_dir is None:
        output_dir = Path("./focused_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Also create a token analysis report
    token_analysis = {}
    
    for seq_type in seq_types:
        print(f"\n📊 Creating {plot_type} plot for {seq_type}...")
        
        # Calculate grid size based on number of important heads
        n_heads = len(important_heads)
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_heads == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each important head
        for idx, (layer_num, head_idx) in enumerate(important_heads):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Get data for this layer
            layer_data = cycle_data.get(layer_num, {}).get(seq_type, {})
            
            if plot_type == 'focus':
                # Plot focus token evolution
                focus_data = layer_data.get('focus_tokens', [])
                if focus_data and head_idx < len(focus_data[0]) if focus_data else False:
                    
                    # Collect token frequencies across sequences for this head
                    token_counts_by_cycle = defaultdict(lambda: defaultdict(int))
                    
                    for seq_focus_tokens in focus_data:
                        if head_idx < len(seq_focus_tokens):
                            head_tokens = seq_focus_tokens[head_idx]
                            for token_info in head_tokens:
                                cycle = token_info['cycle']
                                token = token_info['token']
                                token_counts_by_cycle[cycle][token] += 1
                    
                    # Plot most common tokens per cycle
                    cycles = sorted(token_counts_by_cycle.keys())
                    colors = ['red', 'blue', 'green']  # Different colors for rank 0, 1, 2
                    
                    # Collect all unique tokens for this head to show in title
                    all_tokens = set()
                    for cycle in cycles:
                        token_counts = token_counts_by_cycle[cycle]
                        most_common = Counter(token_counts).most_common(3)
                        for token, count in most_common:
                            all_tokens.add(token)
                    
                    for cycle in cycles:
                        token_counts = token_counts_by_cycle[cycle]
                        most_common = Counter(token_counts).most_common(3)
                        
                        for i, (token, count) in enumerate(most_common):
                            # Create label with token name and count
                            label = f"Rank {i}: {token} ({count})" if cycle == cycles[0] else ""
                            ax.scatter(cycle, i, s=count*20, alpha=0.7, 
                                     color=colors[i % len(colors)], label=label)
                            
                            # Annotate with token name
                            ax.annotate(token, (cycle, i), xytext=(5, 5), 
                                      textcoords='offset points', fontsize=8, alpha=0.8)
                    
                    # Show most common tokens in title
                    top_tokens = list(all_tokens)[:3]  # Show up to 3 most common
                    tokens_str = ', '.join([f'"{t}"' for t in top_tokens])
                    ax.set_title(f'L{layer_num}H{head_idx}\nTokens: {tokens_str}', fontsize=10)
                    ax.set_xlabel('Cycle')
                    ax.set_ylabel('Token Rank')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend if there are labels
                    if cycles:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    
            elif plot_type == 'attention':
                # Plot attention distribution evolution
                attention_data = layer_data.get('attention_distributions', [])
                if attention_data and head_idx < len(attention_data[0]) if attention_data else False:
                    
                    # Aggregate attention distributions
                    cycle_distributions = defaultdict(lambda: defaultdict(list))
                    
                    for seq_distributions in attention_data:
                        if head_idx < len(seq_distributions):
                            head_distributions = seq_distributions[head_idx]
                            for dist_info in head_distributions:
                                cycle_end = dist_info['cycle_end']
                                segments = dist_info['segments']
                                
                                # Store prompt attention
                                cycle_distributions[cycle_end]['prompt'].append(segments['prompt'])
                                
                                # Store cycle attentions
                                for cycle_name, attention in segments['cycles'].items():
                                    cycle_distributions[cycle_end][cycle_name].append(attention)
                    
                    # Plot mean attention distributions
                    cycles = sorted(cycle_distributions.keys())
                    segment_names = ['prompt'] + [f'cycle_{i+1}' for i in range(4)]  # 4 cycles
                    
                    for segment_name in segment_names:
                        means = []
                        stds = []
                        for cycle in cycles:
                            values = cycle_distributions[cycle][segment_name]
                            if values:
                                means.append(np.mean(values))
                                stds.append(np.std(values))
                            else:
                                means.append(0)
                                stds.append(0)
                        
                        ax.errorbar(cycles, means, yerr=stds, label=segment_name, 
                                  marker='o', alpha=0.7)
                    
                    ax.set_title(f'L{layer_num}H{head_idx} Attention Evolution')
                    ax.set_xlabel('Cycle End')
                    ax.set_ylabel('Attention Weight')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(important_heads), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{plot_type.title()} Evolution - {seq_type} (Important Heads Only)')
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"{seq_type}_{plot_type}_evolution_important_heads.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved plot: {plot_path}")
        
        # Generate token analysis report
        if plot_type == 'focus':
            generate_token_analysis_report(cycle_data, important_heads, seq_type, output_dir)

def generate_token_analysis_report(cycle_data, important_heads, seq_type, output_dir):
    """Generate a detailed report of token focus patterns for each head."""
    
    print(f"📝 Generating token analysis report for {seq_type}...")
    
    report_lines = []
    report_lines.append(f"# Token Focus Analysis - {seq_type.title()}")
    report_lines.append(f"Analysis of which tokens each important head focuses on most frequently.")
    report_lines.append("")
    
    for layer_num, head_idx in important_heads:
        layer_data = cycle_data.get(layer_num, {}).get(seq_type, {})
        focus_data = layer_data.get('focus_tokens', [])
        
        if not focus_data or head_idx >= len(focus_data[0]) if focus_data else True:
            continue
            
        report_lines.append(f"## Layer {layer_num}, Head {head_idx}")
        report_lines.append("")
        
        # Collect all tokens this head focuses on across all sequences and cycles
        all_tokens = defaultdict(int)
        token_by_cycle = defaultdict(lambda: defaultdict(int))
        
        for seq_focus_tokens in focus_data:
            if head_idx < len(seq_focus_tokens):
                head_tokens = seq_focus_tokens[head_idx]
                for token_info in head_tokens:
                    cycle = token_info['cycle']
                    token = token_info['token']
                    all_tokens[token] += 1
                    token_by_cycle[cycle][token] += 1
        
        # Overall most common tokens
        most_common_overall = Counter(all_tokens).most_common(10)
        report_lines.append("**Most Focused Tokens (Overall):**")
        for rank, (token, count) in enumerate(most_common_overall, 1):
            percentage = (count / sum(all_tokens.values())) * 100
            report_lines.append(f"{rank}. `{token}` - {count} times ({percentage:.1f}%)")
        report_lines.append("")
        
        # Per-cycle analysis
        report_lines.append("**Per-Cycle Focus:**")
        for cycle in sorted(token_by_cycle.keys()):
            cycle_tokens = Counter(token_by_cycle[cycle]).most_common(3)
            tokens_str = ", ".join([f"`{token}` ({count})" for token, count in cycle_tokens])
            report_lines.append(f"- **Cycle {cycle}**: {tokens_str}")
        report_lines.append("")
        
        # Consistency analysis
        total_sequences = len(focus_data)
        consistency_scores = {}
        for token, count in most_common_overall:
            consistency_scores[token] = count / total_sequences
        
        most_consistent = max(consistency_scores.items(), key=lambda x: x[1]) if consistency_scores else ("None", 0)
        report_lines.append(f"**Most Consistent Token**: `{most_consistent[0]}` (appears in {most_consistent[1]:.1%} of sequences)")
        report_lines.append("")
    
    # Save report
    report_path = output_dir / f"{seq_type}_token_analysis.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   ✅ Token analysis saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Create focused plots for important attention heads")
    parser.add_argument("--contrast_path", type=str, 
                       default="/home/mmahaut/projects/parrots/outputs_multihead_full/EleutherAI/pythia-1.4b/steplatest",
                       help="Path to contrast data logs")
    parser.add_argument("--cycle_data_path", type=str,
                       default="/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/cycle_evolution_parametric/cycles_4/steplatest",
                       help="Path to cycle evolution data")
    parser.add_argument("--output_dir", type=str, default="./focused_important_heads_plots",
                       help="Output directory for plots")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top heads to focus on")
    parser.add_argument("--cycle_size", type=int, default=3, help="Cycle size for contrast extraction")
    parser.add_argument("--t_size", type=int, default=32, help="Template size for contrast extraction")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Starting focused head analysis...")
    print(f"   - Contrast path: {args.contrast_path}")
    print(f"   - Cycle data path: {args.cycle_data_path}")
    print(f"   - Output directory: {output_dir}")
    print(f"   - Top K heads: {args.top_k}")
    
    # Step 1: Extract contrast values to identify important heads
    contrast_data = extract_contrast_values(args.contrast_path, args.cycle_size, args.t_size)
    
    if not contrast_data:
        print("❌ No contrast data found!")
        return
    
    # Step 2: Identify top K important heads
    important_heads = identify_important_heads(contrast_data, args.top_k)
    
    # Step 3: Load cycle evolution data
    cycle_data = load_cycle_evolution_data(args.cycle_data_path)
    
    if not cycle_data:
        print("❌ No cycle evolution data found!")
        return
    
    # Step 4: Create focused plots for natural and no_cycle_icl
    seq_types = ['natural', 'no_cycle_icl']
    
    for plot_type in ['focus', 'attention']:
        plot_focused_evolution(
            cycle_data, 
            important_heads, 
            seq_types=seq_types,
            plot_type=plot_type,
            output_dir=output_dir
        )
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Created plots for {len(important_heads)} important heads")

if __name__ == "__main__":
    main()