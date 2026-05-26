#!/usr/bin/env python3

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

def aggregate_parallel_results():
    """
    Aggregate results from parallel parametric analysis.
    Creates comprehensive comparison plots across all cycle/layer combinations.
    """
    
    print("🔄 Aggregating Parallel Cycle Evolution Results")
    
    # Find all result files
    results_dir = Path("../plots/cycle_evolution_parametric")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Collect all results
    all_results = {}
    summary_stats = {}
    
    print("📁 Scanning for result files...")
    
    for cycles_dir in results_dir.glob("cycles_*"):
        cycles = int(cycles_dir.name.split("_")[1])
        
        for layer_dir in cycles_dir.glob("layer_*"):
            layer = int(layer_dir.name.split("_")[1])
            
            # Load results
            result_file = layer_dir / f"results_c{cycles}_l{layer}.pt"
            summary_file = layer_dir / f"summary_c{cycles}_l{layer}.json"
            
            if result_file.exists():
                print(f"   📊 Loading cycles={cycles}, layer={layer}")
                results = torch.load(result_file)
                all_results[(cycles, layer)] = results
                
                # Load summary stats
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary_stats[(cycles, layer)] = json.load(f)
    
    if not all_results:
        print("❌ No result files found!")
        return
    
    print(f"✅ Found {len(all_results)} parameter combinations")
    
    # Create aggregated output directory
    agg_output_dir = Path("../plots/aggregated_cycle_evolution")
    agg_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract consistency metrics for all combinations
    consistency_data = defaultdict(lambda: defaultdict(list))
    
    for (cycles, layer), results in all_results.items():
        for seq_type, evolution_data in results.items():
            if 'sequences' in evolution_data:
                for seq_data in evolution_data['sequences']:
                    for layer_name, layer_data in seq_data['layer_results'].items():
                        for head_name, evolution_matrix in layer_data.items():
                            if evolution_matrix.shape[0] >= 2:
                                # Consistency metric: correlation between first and last cycle
                                first_cycle = evolution_matrix[0].flatten()
                                last_cycle = evolution_matrix[-1].flatten()
                                corr = np.corrcoef(first_cycle, last_cycle)[0, 1]
                                if not np.isnan(corr):
                                    consistency_data[seq_type][(cycles, layer)].append(corr)
    
    # Create comprehensive comparison plots
    create_parameter_heatmaps(consistency_data, agg_output_dir)
    create_cycle_layer_comparison(consistency_data, agg_output_dir)
    create_sequence_type_comparison(consistency_data, agg_output_dir)
    
    # Create summary report
    create_summary_report(all_results, summary_stats, agg_output_dir)
    
    print(f"\n🎉 Aggregation Complete!")
    print(f"📁 Aggregated results saved to: {agg_output_dir}")
    print(f"🔍 Key files to check:")
    print(f"  - parameter_heatmaps.png: Consistency across all parameter combinations")
    print(f"  - cycle_layer_comparison.png: How cycles and layers affect consistency")
    print(f"  - sequence_type_comparison.png: Differences between natural/ICL/no-cycle")
    print(f"  - summary_report.json: Detailed statistics")

def create_parameter_heatmaps(consistency_data, output_dir):
    """Create heatmaps showing consistency across cycle/layer combinations."""
    
    print("📊 Creating parameter heatmaps...")
    
    # Get all unique cycles and layers
    all_params = set()
    for seq_type_data in consistency_data.values():
        all_params.update(seq_type_data.keys())
    
    cycles = sorted(set(c for c, l in all_params))
    layers = sorted(set(l for c, l in all_params))
    
    # Create heatmap for each sequence type
    fig, axes = plt.subplots(1, len(consistency_data), figsize=(5*len(consistency_data), 4))
    if len(consistency_data) == 1:
        axes = [axes]
    
    for idx, (seq_type, param_data) in enumerate(consistency_data.items()):
        # Create matrix
        matrix = np.full((len(cycles), len(layers)), np.nan)
        
        for i, c in enumerate(cycles):
            for j, l in enumerate(layers):
                if (c, l) in param_data and param_data[(c, l)]:
                    matrix[i, j] = np.mean(param_data[(c, l)])
        
        # Plot heatmap
        im = axes[idx].imshow(matrix, aspect='auto', cmap='RdYlBu_r', vmin=-1, vmax=1)
        axes[idx].set_title(f'{seq_type.title()} Sequences\nAttention Consistency')
        axes[idx].set_xlabel('Layer')
        axes[idx].set_ylabel('Cycles')
        axes[idx].set_xticks(range(len(layers)))
        axes[idx].set_xticklabels(layers)
        axes[idx].set_yticks(range(len(cycles)))
        axes[idx].set_yticklabels(cycles)
        
        # Add text annotations
        for i in range(len(cycles)):
            for j in range(len(layers)):
                if not np.isnan(matrix[i, j]):
                    axes[idx].text(j, i, f'{matrix[i, j]:.2f}', 
                                 ha='center', va='center', 
                                 color='white' if abs(matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=axes, label='First-Last Cycle Correlation')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cycle_layer_comparison(consistency_data, output_dir):
    """Create plots comparing consistency across cycles and layers."""
    
    print("📊 Creating cycle/layer comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Consistency by cycles (averaged across layers)
    for seq_type, param_data in consistency_data.items():
        cycles_avg = defaultdict(list)
        for (c, l), consistencies in param_data.items():
            cycles_avg[c].extend(consistencies)
        
        cycles = sorted(cycles_avg.keys())
        means = [np.mean(cycles_avg[c]) for c in cycles]
        stds = [np.std(cycles_avg[c]) for c in cycles]
        
        axes[0,0].errorbar(cycles, means, yerr=stds, marker='o', label=seq_type, capsize=5)
    
    axes[0,0].set_title('Attention Consistency by Number of Cycles')
    axes[0,0].set_xlabel('Number of Cycles')
    axes[0,0].set_ylabel('Consistency (Correlation)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Consistency by layers (averaged across cycles)
    for seq_type, param_data in consistency_data.items():
        layers_avg = defaultdict(list)
        for (c, l), consistencies in param_data.items():
            layers_avg[l].extend(consistencies)
        
        layers = sorted(layers_avg.keys())
        means = [np.mean(layers_avg[l]) for l in layers]
        stds = [np.std(layers_avg[l]) for l in layers]
        
        axes[0,1].errorbar(layers, means, yerr=stds, marker='s', label=seq_type, capsize=5)
    
    axes[0,1].set_title('Attention Consistency by Layer')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Consistency (Correlation)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison
    all_consistencies = []
    seq_labels = []
    
    for seq_type, param_data in consistency_data.items():
        consistencies = []
        for param_consistencies in param_data.values():
            consistencies.extend(param_consistencies)
        if consistencies:
            all_consistencies.append(consistencies)
            seq_labels.append(seq_type)
    
    axes[1,0].boxplot(all_consistencies, labels=seq_labels)
    axes[1,0].set_title('Consistency Distribution by Sequence Type')
    axes[1,0].set_ylabel('Consistency (Correlation)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Parameter interaction
    for seq_type, param_data in consistency_data.items():
        cycles_list = []
        layers_list = []
        consistency_list = []
        
        for (c, l), consistencies in param_data.items():
            for cons in consistencies:
                cycles_list.append(c)
                layers_list.append(l)
                consistency_list.append(cons)
        
        if consistency_list:
            scatter = axes[1,1].scatter(cycles_list, layers_list, 
                                      c=consistency_list, s=50, alpha=0.6,
                                      cmap='RdYlBu_r', vmin=-1, vmax=1, label=seq_type)
    
    axes[1,1].set_title('Parameter Interaction (Color = Consistency)')
    axes[1,1].set_xlabel('Cycles')
    axes[1,1].set_ylabel('Layer')
    plt.colorbar(scatter, ax=axes[1,1], label='Consistency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cycle_layer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sequence_type_comparison(consistency_data, output_dir):
    """Create detailed comparison between sequence types."""
    
    print("📊 Creating sequence type comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Statistical comparison
    seq_types = list(consistency_data.keys())
    
    # Collect all consistency values for each sequence type
    all_seq_consistencies = {}
    for seq_type, param_data in consistency_data.items():
        consistencies = []
        for param_consistencies in param_data.values():
            consistencies.extend(param_consistencies)
        all_seq_consistencies[seq_type] = consistencies
    
    # Plot distributions
    axes[0,0].hist([all_seq_consistencies[st] for st in seq_types], 
                  bins=30, alpha=0.7, label=seq_types)
    axes[0,0].set_title('Consistency Distributions by Sequence Type')
    axes[0,0].set_xlabel('Consistency (Correlation)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].legend()
    
    # Box plot
    axes[0,1].boxplot([all_seq_consistencies[st] for st in seq_types], labels=seq_types)
    axes[0,1].set_title('Consistency Comparison')
    axes[0,1].set_ylabel('Consistency (Correlation)')
    
    # Statistical summary
    summary_text = "Statistical Summary:\n\n"
    for seq_type, consistencies in all_seq_consistencies.items():
        if consistencies:
            mean_cons = np.mean(consistencies)
            std_cons = np.std(consistencies)
            summary_text += f"{seq_type}:\n"
            summary_text += f"  Mean: {mean_cons:.3f} ± {std_cons:.3f}\n"
            summary_text += f"  N: {len(consistencies)}\n\n"
    
    axes[1,0].text(0.1, 0.9, summary_text, transform=axes[1,0].transAxes, 
                  verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1,0].set_title('Summary Statistics')
    axes[1,0].axis('off')
    
    # Interpretation guide
    interp_text = """Interpretation Guide:

Consistency Score (Correlation):
• > 0.7: Highly consistent attention
• 0.3-0.7: Moderately consistent  
• < 0.3: Attention changes significantly

Sequence Types:
• Natural: Model chose to repeat
• ICL: We forced repetition  
• No-cycle ICL: Never naturally repeated

Key Questions:
1. Do natural cycles show higher consistency?
2. Do forced repetitions (ICL) behave differently?
3. Which layers/cycles show most consistency?"""
    
    axes[1,1].text(0.1, 0.9, interp_text, transform=axes[1,1].transAxes,
                  verticalalignment='top', fontsize=10)
    axes[1,1].set_title('How to Interpret Results')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sequence_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(all_results, summary_stats, output_dir):
    """Create a comprehensive summary report."""
    
    report = {
        'total_parameter_combinations': len(all_results),
        'parameters_tested': {
            'cycles': sorted(set(c for c, l in all_results.keys())),
            'layers': sorted(set(l for c, l in all_results.keys()))
        },
        'sequence_types_found': {},
        'overall_statistics': {}
    }
    
    # Count sequences by type
    seq_type_counts = defaultdict(int)
    for results in all_results.values():
        for seq_type, evolution_data in results.items():
            if 'sequences' in evolution_data:
                seq_type_counts[seq_type] += len(evolution_data['sequences'])
    
    report['sequence_types_found'] = dict(seq_type_counts)
    
    # Save report
    with open(output_dir / 'summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Summary report saved")

if __name__ == "__main__":
    aggregate_parallel_results()