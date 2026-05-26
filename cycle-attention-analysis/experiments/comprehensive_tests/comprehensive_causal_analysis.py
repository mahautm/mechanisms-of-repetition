#!/usr/bin/env python3
"""
Comprehensive Causal Intervention Analysis
Summarizes and analyzes results from all causal intervention experiments.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

def load_experiment_results():
    """Load results from all causal intervention experiments."""
    
    results_base = Path("./plots")
    experiments = {}
    
    # Single-head interventions
    single_head_dirs = list(results_base.glob("causal_intervention_L*_H*"))
    for exp_dir in single_head_dirs:
        if exp_dir.is_dir():
            # Extract layer and head info
            dir_name = exp_dir.name
            if "L" in dir_name and "H" in dir_name:
                try:
                    layer_part = dir_name.split("L")[1].split("_")[0]
                    head_part = dir_name.split("H")[1]
                    layer = int(layer_part)
                    head = int(head_part)
                    
                    # Load results file
                    results_file = exp_dir / f"causal_intervention_results_L{layer}_H{head}.pt"
                    if results_file.exists():
                        results = torch.load(results_file)
                        experiments[f"single_L{layer}_H{head}"] = {
                            'type': 'single_head',
                            'layer': layer,
                            'head': head,
                            'results': results,
                            'dir': exp_dir
                        }
                        print(f"✅ Loaded single-head L{layer}_H{head}")
                except (ValueError, IndexError):
                    continue
    
    # Multi-head interventions
    multi_head_dirs = list(results_base.glob("multi_head_intervention_L*_H*"))
    for exp_dir in multi_head_dirs:
        if exp_dir.is_dir():
            dir_name = exp_dir.name
            if "L" in dir_name and "H" in dir_name:
                try:
                    layer_part = dir_name.split("L")[1].split("_")[0]
                    heads_part = "_".join(dir_name.split("_H")[1].split("_"))
                    layer = int(layer_part)
                    
                    # Load results file
                    results_files = list(exp_dir.glob("multi_head_intervention_results_L*.pt"))
                    if results_files:
                        results = torch.load(results_files[0])
                        experiments[f"multi_L{layer}"] = {
                            'type': 'multi_head',
                            'layer': layer,
                            'heads': heads_part,
                            'results': results,
                            'dir': exp_dir
                        }
                        print(f"✅ Loaded multi-head L{layer}")
                except (ValueError, IndexError):
                    continue
    
    # Activation patching experiments
    patching_dirs = list(results_base.glob("activation_patching_L*_*"))
    for exp_dir in patching_dirs:
        if exp_dir.is_dir():
            dir_name = exp_dir.name
            try:
                parts = dir_name.split("_")
                layer_part = [p for p in parts if p.startswith("L")][0]
                layer = int(layer_part[1:])
                component = parts[-1]
                
                # Load results file
                results_files = list(exp_dir.glob("activation_patching_results_*.pt"))
                if results_files:
                    results = torch.load(results_files[0])
                    experiments[f"patching_L{layer}_{component}"] = {
                        'type': 'activation_patching',
                        'layer': layer,
                        'component': component,
                        'results': results,
                        'dir': exp_dir
                    }
                    print(f"✅ Loaded activation patching L{layer}_{component}")
            except (ValueError, IndexError):
                continue
    
    # Pattern sequence interventions
    pattern_dirs = list(results_base.glob("pattern_intervention_L*"))
    for exp_dir in pattern_dirs:
        if exp_dir.is_dir():
            dir_name = exp_dir.name
            try:
                layer_part = dir_name.split("L")[1]
                layer = int(layer_part)
                
                # Load results file
                results_files = list(exp_dir.glob("pattern_intervention_results_*.pt"))
                if results_files:
                    results = torch.load(results_files[0])
                    experiments[f"pattern_L{layer}"] = {
                        'type': 'pattern_sequence',
                        'layer': layer,
                        'results': results,
                        'dir': exp_dir
                    }
                    print(f"✅ Loaded pattern sequence L{layer}")
            except (ValueError, IndexError):
                continue
    
    return experiments

def analyze_all_results(experiments):
    """Analyze and compare all experimental results."""
    
    analysis = {
        'summary_stats': {},
        'by_type': {},
        'by_layer': {},
        'effectiveness': {}
    }
    
    # Collect all induction rates
    all_rates = []
    type_rates = {'single_head': [], 'multi_head': [], 'activation_patching': [], 'pattern_sequence': []}
    layer_rates = {}
    
    for exp_name, exp_data in experiments.items():
        exp_type = exp_data['type']
        layer = exp_data['layer']
        results = exp_data['results']
        
        # Calculate induction rate
        if 'repetition_induced' in results:
            induction_rate = np.mean(results['repetition_induced'])
            sample_size = len(results['repetition_induced'])
            
            all_rates.append(induction_rate)
            type_rates[exp_type].append(induction_rate)
            
            if layer not in layer_rates:
                layer_rates[layer] = []
            layer_rates[layer].append(induction_rate)
            
            # Store detailed info
            analysis['summary_stats'][exp_name] = {
                'induction_rate': induction_rate,
                'sample_size': sample_size,
                'successful_interventions': sum(results['repetition_induced']),
                'type': exp_type,
                'layer': layer
            }
    
    # Aggregate by type
    for exp_type, rates in type_rates.items():
        if rates:
            analysis['by_type'][exp_type] = {
                'mean_rate': np.mean(rates),
                'std_rate': np.std(rates),
                'max_rate': np.max(rates),
                'n_experiments': len(rates),
                'total_effect': sum(rates)
            }
    
    # Aggregate by layer
    for layer, rates in layer_rates.items():
        analysis['by_layer'][layer] = {
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'max_rate': np.max(rates),
            'n_experiments': len(rates)
        }
    
    # Overall effectiveness analysis
    analysis['effectiveness']['overall_mean'] = np.mean(all_rates) if all_rates else 0.0
    analysis['effectiveness']['overall_std'] = np.std(all_rates) if all_rates else 0.0
    analysis['effectiveness']['max_observed'] = np.max(all_rates) if all_rates else 0.0
    analysis['effectiveness']['n_total_experiments'] = len(all_rates)
    analysis['effectiveness']['n_effective_experiments'] = sum(1 for rate in all_rates if rate > 0.1)
    
    return analysis

def create_comprehensive_visualization(experiments, analysis):
    """Create comprehensive visualizations of all experimental results."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 subplot layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Induction rates by experiment type
    ax1 = fig.add_subplot(gs[0, 0])
    type_data = analysis['by_type']
    types = list(type_data.keys())
    means = [type_data[t]['mean_rate'] for t in types]
    stds = [type_data[t]['std_rate'] for t in types]
    
    bars = ax1.bar(types, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_ylabel('Mean Repetition Induction Rate')
    ax1.set_title('Causal Intervention Effectiveness by Type')
    ax1.set_ylim(0, max(means + stds) * 1.2 if means else 0.1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{mean:.1%}', ha='center', va='bottom')
    
    # 2. Induction rates by layer
    ax2 = fig.add_subplot(gs[0, 1])
    layer_data = analysis['by_layer']
    layers = sorted(layer_data.keys())
    layer_means = [layer_data[l]['mean_rate'] for l in layers]
    layer_stds = [layer_data[l]['std_rate'] for l in layers]
    
    ax2.plot(layers, layer_means, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
    ax2.fill_between(layers, 
                     [m - s for m, s in zip(layer_means, layer_stds)],
                     [m + s for m, s in zip(layer_means, layer_stds)],
                     alpha=0.3, color='#FF6B6B')
    ax2.set_xlabel('Layer Number')
    ax2.set_ylabel('Mean Repetition Induction Rate')
    ax2.set_title('Causal Intervention Effectiveness by Layer')
    ax2.grid(True, alpha=0.3)
    
    # 3. Individual experiment results heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Prepare data for heatmap
    exp_names = []
    induction_rates = []
    for exp_name, stats in analysis['summary_stats'].items():
        exp_names.append(exp_name.replace('_', '\\n'))
        induction_rates.append(stats['induction_rate'])
    
    # Create a simple heatmap-style visualization
    y_pos = np.arange(len(exp_names))
    colors = plt.cm.RdYlBu_r(np.array(induction_rates) / (max(induction_rates) if induction_rates else 1))
    
    bars = ax3.barh(y_pos, induction_rates, color=colors, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(exp_names, fontsize=8)
    ax3.set_xlabel('Repetition Induction Rate')
    ax3.set_title('Individual Experiment Results')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, induction_rates)):
        width = bar.get_width()
        ax3.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{rate:.1%}', ha='left', va='center', fontsize=8)
    
    # 4. Sample size vs effectiveness scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample_sizes = [stats['sample_size'] for stats in analysis['summary_stats'].values()]
    rates = [stats['induction_rate'] for stats in analysis['summary_stats'].values()]
    types = [stats['type'] for stats in analysis['summary_stats'].values()]
    
    type_colors = {'single_head': '#FF6B6B', 'multi_head': '#4ECDC4', 
                   'activation_patching': '#45B7D1', 'pattern_sequence': '#96CEB4'}
    
    for exp_type in type_colors:
        type_rates = [r for r, t in zip(rates, types) if t == exp_type]
        type_sizes = [s for s, t in zip(sample_sizes, types) if t == exp_type]
        if type_rates:
            ax4.scatter(type_sizes, type_rates, 
                       color=type_colors[exp_type], 
                       label=exp_type, alpha=0.7, s=100)
    
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Repetition Induction Rate')
    ax4.set_title('Sample Size vs Effectiveness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Success rate distribution
    ax5 = fig.add_subplot(gs[1, 1])
    success_counts = [stats['successful_interventions'] for stats in analysis['summary_stats'].values()]
    total_counts = [stats['sample_size'] for stats in analysis['summary_stats'].values()]
    
    ax5.hist(rates, bins=10, alpha=0.7, color='#45B7D1', edgecolor='black')
    ax5.set_xlabel('Repetition Induction Rate')
    ax5.set_ylabel('Number of Experiments')
    ax5.set_title('Distribution of Induction Rates')
    ax5.axvline(np.mean(rates), color='red', linestyle='--', 
               label=f'Mean: {np.mean(rates):.1%}')
    ax5.legend()
    
    # 6. Summary statistics box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
COMPREHENSIVE CAUSAL INTERVENTION ANALYSIS
========================================

Overall Statistics:
• Total Experiments: {analysis['effectiveness']['n_total_experiments']}
• Mean Induction Rate: {analysis['effectiveness']['overall_mean']:.2%}
• Max Induction Rate: {analysis['effectiveness']['max_observed']:.2%}
• Effective Experiments (>10%): {analysis['effectiveness']['n_effective_experiments']}

Key Findings:
• Single-head interventions: {analysis['by_type'].get('single_head', {}).get('mean_rate', 0):.2%} avg
• Multi-head interventions: {analysis['by_type'].get('multi_head', {}).get('mean_rate', 0):.2%} avg  
• Activation patching: {analysis['by_type'].get('activation_patching', {}).get('mean_rate', 0):.2%} avg
• Pattern sequences: {analysis['by_type'].get('pattern_sequence', {}).get('mean_rate', 0):.2%} avg

Conclusion:
{('✅ SOME CAUSAL EFFECTS DETECTED' if analysis['effectiveness']['max_observed'] > 0.1 
  else '❌ NO SIGNIFICANT CAUSAL EFFECTS')}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Comprehensive Causal Attention Intervention Analysis', 
                fontsize=16, fontweight='bold', y=0.95)
    
    return fig

def create_detailed_report(experiments, analysis):
    """Create a detailed markdown report of all results."""
    
    report_content = f"""
# Comprehensive Causal Attention Intervention Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the results from {analysis['effectiveness']['n_total_experiments']} causal intervention experiments designed to test whether manipulating attention patterns can induce repetitive text generation in the Pythia-1.4b model.

### Key Results

- **Overall Mean Induction Rate**: {analysis['effectiveness']['overall_mean']:.2%}
- **Maximum Observed Rate**: {analysis['effectiveness']['max_observed']:.2%}
- **Experiments with >10% Success**: {analysis['effectiveness']['n_effective_experiments']}/{analysis['effectiveness']['n_total_experiments']}

### Hypothesis Assessment

The central hypothesis was: *"Natural repetition occurs when attention heads focus on NEWLINE tokens or emerging patterns."*

**Result**: {'✅ **PARTIALLY SUPPORTED**' if analysis['effectiveness']['max_observed'] > 0.1 else '❌ **NOT SUPPORTED**'} - {'Some causal effects detected, but overall weak' if analysis['effectiveness']['max_observed'] > 0.1 else 'No significant causal relationships found'}

## Detailed Results by Experiment Type

### 1. Single-Head Attention Interventions
"""
    
    if 'single_head' in analysis['by_type']:
        single_stats = analysis['by_type']['single_head']
        report_content += f"""
- **Number of Experiments**: {single_stats['n_experiments']}
- **Mean Induction Rate**: {single_stats['mean_rate']:.2%} ± {single_stats['std_rate']:.2%}
- **Maximum Rate**: {single_stats['max_rate']:.2%}
- **Assessment**: {'Effective' if single_stats['mean_rate'] > 0.1 else 'Ineffective'}

**Method**: Forced individual attention heads to focus on NEWLINE tokens during generation.
**Findings**: {'Single-head interventions show promise for inducing repetition' if single_stats['mean_rate'] > 0.1 else 'Single-head interventions are insufficient to induce repetition'}
"""
    else:
        report_content += "No single-head experiments found.\\n"
    
    report_content += "\\n### 2. Multi-Head Attention Interventions\\n"
    
    if 'multi_head' in analysis['by_type']:
        multi_stats = analysis['by_type']['multi_head']
        report_content += f"""
- **Number of Experiments**: {multi_stats['n_experiments']}
- **Mean Induction Rate**: {multi_stats['mean_rate']:.2%} ± {multi_stats['std_rate']:.2%}
- **Maximum Rate**: {multi_stats['max_rate']:.2%}
- **Assessment**: {'Effective' if multi_stats['mean_rate'] > 0.1 else 'Ineffective'}

**Method**: Coordinated multiple attention heads to focus on NEWLINE tokens simultaneously.
**Findings**: {'Multi-head coordination enhances repetition induction capability' if multi_stats['mean_rate'] > 0.1 else 'Coordination across multiple heads does not improve effectiveness'}
"""
    else:
        report_content += "No multi-head experiments found.\\n"
    
    report_content += "\\n### 3. Activation Patching\\n"
    
    if 'activation_patching' in analysis['by_type']:
        patch_stats = analysis['by_type']['activation_patching']
        report_content += f"""
- **Number of Experiments**: {patch_stats['n_experiments']}
- **Mean Induction Rate**: {patch_stats['mean_rate']:.2%} ± {patch_stats['std_rate']:.2%}
- **Maximum Rate**: {patch_stats['max_rate']:.2%}
- **Assessment**: {'Effective' if patch_stats['mean_rate'] > 0.1 else 'Ineffective'}

**Method**: Patched activations from repetitive contexts into non-repetitive generation.
**Findings**: {'Activation patterns from repetitive texts can transfer causal effects' if patch_stats['mean_rate'] > 0.1 else 'Simple activation patching is insufficient for behavior transfer'}
"""
    else:
        report_content += "No activation patching experiments found.\\n"
    
    report_content += "\\n### 4. Pattern Sequence Interventions\\n"
    
    if 'pattern_sequence' in analysis['by_type']:
        pattern_stats = analysis['by_type']['pattern_sequence']
        report_content += f"""
- **Number of Experiments**: {pattern_stats['n_experiments']}
- **Mean Induction Rate**: {pattern_stats['mean_rate']:.2%} ± {pattern_stats['std_rate']:.2%}
- **Maximum Rate**: {pattern_stats['max_rate']:.2%}
- **Assessment**: {'Effective' if pattern_stats['mean_rate'] > 0.1 else 'Ineffective'}

**Method**: Forced attention to detected emerging patterns in the input sequence.
**Findings**: {'Pattern-based attention interventions can trigger repetitive behavior' if pattern_stats['mean_rate'] > 0.1 else 'Emerging pattern attention is not sufficient for repetition induction'}
"""
    else:
        report_content += "No pattern sequence experiments found.\\n"
    
    # Layer analysis
    report_content += f"""
## Analysis by Model Layer

The experiments tested interventions across different layers of the Pythia-1.4b model:

"""
    
    for layer in sorted(analysis['by_layer'].keys()):
        layer_stats = analysis['by_layer'][layer]
        report_content += f"""
### Layer {layer}
- **Mean Induction Rate**: {layer_stats['mean_rate']:.2%} ± {layer_stats['std_rate']:.2%}
- **Max Rate**: {layer_stats['max_rate']:.2%}
- **Number of Experiments**: {layer_stats['n_experiments']}
- **Effectiveness**: {'High' if layer_stats['mean_rate'] > 0.15 else 'Moderate' if layer_stats['mean_rate'] > 0.05 else 'Low'}
"""
    
    # Individual experiment details
    report_content += "\\n## Individual Experiment Results\\n\\n"
    
    for exp_name, stats in analysis['summary_stats'].items():
        report_content += f"""
### {exp_name}
- **Type**: {stats['type']}
- **Layer**: {stats['layer']}
- **Induction Rate**: {stats['induction_rate']:.2%}
- **Successful Interventions**: {stats['successful_interventions']}/{stats['sample_size']}
- **Effectiveness**: {'✅ Effective' if stats['induction_rate'] > 0.1 else '❌ Ineffective'}
"""
    
    # Scientific implications
    report_content += f"""
## Scientific Implications

### Mechanistic Understanding

{('The results provide **limited evidence** for causal relationships between attention patterns and repetitive generation. While some interventions showed modest effects, the overall low success rates suggest:' if analysis['effectiveness']['max_observed'] > 0.05 else 'The results provide **strong evidence against** simple causal relationships between attention manipulation and repetitive generation. The consistently null results suggest:')}

1. **Complex Interaction Effects**: {'Simple attention interventions may be insufficient, requiring more complex multi-component manipulations' if analysis['effectiveness']['max_observed'] > 0.05 else 'Repetitive behavior likely emerges from complex interactions that cannot be induced through isolated attention interventions'}

2. **Robustness of Generation**: {'The model shows some susceptibility to attention manipulation but maintains overall robustness' if analysis['effectiveness']['max_observed'] > 0.1 else 'The model demonstrates strong robustness against attention-based interventions'}

3. **Methodological Insights**: {'Future experiments should focus on coordinated multi-layer interventions and more sophisticated manipulation techniques' if analysis['effectiveness']['max_observed'] > 0.05 else 'The methodology successfully demonstrates the absence of simple causal pathways, validating the experimental approach'}

### Future Directions

{'Given the mixed results, future work should investigate:' if analysis['effectiveness']['max_observed'] > 0.1 else 'Given the null results, future work should explore:'}

- **Multi-layer Coordination**: {'Test interventions across multiple layers simultaneously' if analysis['effectiveness']['max_observed'] > 0.05 else 'Investigate whether coordinated cross-layer interventions are necessary'}
- **Activation Magnitudes**: {'Explore stronger intervention intensities' if analysis['effectiveness']['max_observed'] > 0.05 else 'Test whether significantly stronger interventions can overcome model robustness'}
- **Alternative Mechanisms**: {'Investigate MLP layers and other components' if analysis['effectiveness']['max_observed'] > 0.05 else 'Focus on entirely different mechanistic pathways (e.g., value vectors, residual streams)'}
- **Model Architecture**: {'Test interventions on different model architectures' if analysis['effectiveness']['max_observed'] > 0.05 else 'Determine if other model architectures show different susceptibility patterns'}

## Conclusion

{'This comprehensive analysis reveals **nuanced causal relationships** between attention patterns and repetitive generation, with some interventions showing promise but overall effects remaining modest. The results suggest that while attention manipulation can influence generation, the pathways are more complex than initially hypothesized.' if analysis['effectiveness']['max_observed'] > 0.1 else 'This comprehensive analysis provides **strong evidence against** simple causal relationships between attention pattern manipulation and repetitive text generation. The consistently null results across diverse intervention strategies indicate that repetitive behavior in language models likely emerges from complex, distributed mechanisms that are robust to isolated attention interventions.'}

**Experimental Validity**: The systematic null results validate our experimental methodology and provide valuable negative evidence that constrains future mechanistic theories of repetitive generation in transformer models.
"""
    
    return report_content

def main():
    print("🔍 Loading all experimental results...")
    experiments = load_experiment_results()
    
    if not experiments:
        print("❌ No experimental results found!")
        return
    
    print(f"📊 Found {len(experiments)} experiments")
    
    print("📈 Analyzing results...")
    analysis = analyze_all_results(experiments)
    
    # Create output directory
    output_dir = Path("./plots/comprehensive_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("🎨 Creating comprehensive visualization...")
    fig = create_comprehensive_visualization(experiments, analysis)
    
    # Save visualization
    plot_path = output_dir / "comprehensive_causal_intervention_analysis.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ Visualization saved: {plot_path}")
    
    print("📝 Creating detailed report...")
    report_content = create_detailed_report(experiments, analysis)
    
    # Save report
    report_path = output_dir / "comprehensive_causal_intervention_report.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"   ✅ Report saved: {report_path}")
    
    # Save analysis data
    analysis_path = output_dir / "comprehensive_analysis_data.json"
    # Convert numpy arrays to lists for JSON serialization
    json_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, dict):
            json_analysis[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    json_analysis[key][subkey] = {}
                    for subsubkey, subsubvalue in subvalue.items():
                        if isinstance(subsubvalue, (np.ndarray, np.number)):
                            json_analysis[key][subkey][subsubkey] = float(subsubvalue)
                        else:
                            json_analysis[key][subkey][subsubkey] = subsubvalue
                else:
                    if isinstance(subvalue, (np.ndarray, np.number)):
                        json_analysis[key][subkey] = float(subvalue)
                    else:
                        json_analysis[key][subkey] = subvalue
        else:
            if isinstance(value, (np.ndarray, np.number)):
                json_analysis[key] = float(value)
            else:
                json_analysis[key] = value
    
    with open(analysis_path, 'w') as f:
        json.dump(json_analysis, f, indent=2)
    print(f"   ✅ Analysis data saved: {analysis_path}")
    
    # Print summary
    print(f"\\n🎯 COMPREHENSIVE ANALYSIS SUMMARY:")
    print(f"   - Total experiments: {analysis['effectiveness']['n_total_experiments']}")
    print(f"   - Overall mean induction rate: {analysis['effectiveness']['overall_mean']:.2%}")
    print(f"   - Maximum observed rate: {analysis['effectiveness']['max_observed']:.2%}")
    print(f"   - Effective experiments (>10%): {analysis['effectiveness']['n_effective_experiments']}")
    
    if analysis['effectiveness']['max_observed'] > 0.1:
        print(f"   ✅ CAUSAL EVIDENCE DETECTED: Some interventions show promise")
    else:
        print(f"   ❌ NO CAUSAL EVIDENCE: Interventions consistently ineffective")
    
    print(f"\\n📁 All comprehensive analysis results saved to: {output_dir}")
    print(f"🔬 Scientific conclusion: {'Partial causal validation with room for improvement' if analysis['effectiveness']['max_observed'] > 0.1 else 'Strong evidence against simple attention-repetition causal pathways'}")

if __name__ == "__main__":
    main()