#!/usr/bin/env python3
"""
OLMo vs Pythia Comparison Experiment
=====================================

Runs a side-by-side comparison of OLMo and Pythia models on the same sample data.
This allows direct comparison of repetition behaviors and slot-filling performance.

Usage:
    python experiments/compare_olmo_pythia.py --sample-size 30
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.olmo_sample_experiment import OLMoSampleExperiment


class ModelComparison:
    """Compare multiple models on the same experiments"""
    
    def __init__(self, models, output_dir, sample_size=30):
        self.models = models
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
        print("🔬 Model Comparison Experiment")
        print("=" * 60)
        print(f"Models to compare: {len(models)}")
        for name, model_id in models.items():
            print(f"  - {name}: {model_id}")
        print(f"Sample size: {sample_size}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)
    
    def run_experiments(self):
        """Run experiments for all models"""
        for model_name, model_id in self.models.items():
            print(f"\n{'=' * 60}")
            print(f"Running experiments for: {model_name}")
            print(f"{'=' * 60}")
            
            # Create model-specific output directory
            model_output = self.output_dir / model_name.replace('/', '_')
            
            # Run experiment
            experiment = OLMoSampleExperiment(
                model_name=model_id,
                output_dir=str(model_output),
                sample_size=self.sample_size
            )
            
            success = experiment.run_all()
            
            if success:
                # Load results
                with open(model_output / "experiment_results.json", 'r') as f:
                    self.results[model_name] = json.load(f)
            else:
                print(f"⚠️  Failed to complete experiments for {model_name}")
                self.results[model_name] = None
        
        return self.results
    
    def create_comparison_plots(self):
        """Create comparative visualizations"""
        print("\n" + "=" * 60)
        print("Creating comparison visualizations...")
        print("=" * 60)
        
        plots_dir = self.output_dir / "comparison_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Filter out failed experiments
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if len(valid_results) < 2:
            print("⚠️  Need at least 2 successful experiments for comparison")
            return []
        
        plots_created = []
        
        # 1. Slot-filling comparison
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models = list(valid_results.keys())
            metrics = ['direct_follow_accuracy', 'exact_match_accuracy', 'nli_factual_accuracy']
            metric_labels = ['Direct Follow', 'Exact Match', 'NLI Factual']
            
            x = np.arange(len(metric_labels))
            width = 0.35
            
            colors = ['#4A90E2', '#E67E22']
            
            for i, (model_name, color) in enumerate(zip(models, colors)):
                values = [
                    valid_results[model_name]['experiments']['slot_filling']['metrics'][m]
                    for m in metrics
                ]
                offset = width * (i - len(models)/2 + 0.5)
                bars = ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1%}', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_title('Slot-Filling Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels)
            ax.legend(fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plot_path = plots_dir / "slot_filling_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append(str(plot_path))
            print(f"✓ Created: {plot_path.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating slot-filling comparison: {e}")
        
        # 2. Repetition rate comparison
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            models = list(valid_results.keys())
            colors = ['#4A90E2', '#E67E22']
            
            # Repetition rates
            rep_rates = [
                valid_results[model]['experiments']['cycle_detection']['statistics']['repetition_rate']
                for model in models
            ]
            
            bars = ax1.bar(models, rep_rates, color=colors, alpha=0.8)
            ax1.set_ylabel('Repetition Rate', fontsize=12, fontweight='bold')
            ax1.set_title('Repetition Rate Comparison', fontsize=13, fontweight='bold')
            ax1.set_ylim(0, max(rep_rates) * 1.2 if max(rep_rates) > 0 else 0.1)
            ax1.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=11)
            
            # Average cycle lengths
            avg_lengths = []
            for model in models:
                stats = valid_results[model]['experiments']['cycle_detection']['statistics']
                avg_len = stats.get('avg_cycle_length', 0)
                avg_lengths.append(avg_len if avg_len > 0 else 0)
            
            bars = ax2.bar(models, avg_lengths, color=colors, alpha=0.8)
            ax2.set_ylabel('Average Cycle Length (tokens)', fontsize=12, fontweight='bold')
            ax2.set_title('Average Cycle Length Comparison', fontsize=13, fontweight='bold')
            ax2.set_ylim(0, max(avg_lengths) * 1.2 if max(avg_lengths) > 0 else 10)
            ax2.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plot_path = plots_dir / "repetition_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append(str(plot_path))
            print(f"✓ Created: {plot_path.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating repetition comparison: {e}")
        
        # 3. Model architecture comparison
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = list(valid_results.keys())
            
            # Extract architecture info
            layers = [valid_results[model]['model_info']['num_layers'] for model in models]
            heads = [valid_results[model]['model_info']['num_heads'] for model in models]
            hidden = [valid_results[model]['model_info']['hidden_size'] / 1000 for model in models]  # Scale to thousands
            
            x = np.arange(len(models))
            width = 0.25
            
            ax.bar(x - width, layers, width, label='Layers', color='#4A90E2', alpha=0.8)
            ax.bar(x, heads, width, label='Attention Heads', color='#9B59B6', alpha=0.8)
            ax.bar(x + width, hidden, width, label='Hidden Size (K)', color='#E67E22', alpha=0.8)
            
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plot_path = plots_dir / "architecture_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append(str(plot_path))
            print(f"✓ Created: {plot_path.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating architecture comparison: {e}")
        
        print(f"\n✓ Created {len(plots_created)} comparison plots")
        return plots_created
    
    def generate_comparison_report(self):
        """Generate comparative analysis report"""
        print("\n" + "=" * 60)
        print("📊 COMPARISON REPORT")
        print("=" * 60)
        
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        report_lines = [
            "# Model Comparison Report\n",
            f"**Sample Size:** {self.sample_size}",
            f"**Models Compared:** {len(valid_results)}\n",
            "## Models\n"
        ]
        
        for model_name in valid_results.keys():
            info = valid_results[model_name]['model_info']
            report_lines.extend([
                f"### {model_name}\n",
                f"- **Layers:** {info['num_layers']}",
                f"- **Attention Heads:** {info['num_heads']}",
                f"- **Hidden Size:** {info['hidden_size']}",
                f"- **Vocab Size:** {info['vocab_size']}\n"
            ])
        
        report_lines.append("## Performance Comparison\n")
        
        # Create comparison table
        report_lines.append("### Slot-Filling Accuracy\n")
        report_lines.append("| Model | Direct Follow | Exact Match | NLI Factual |")
        report_lines.append("|-------|---------------|-------------|-------------|")
        
        for model_name in valid_results.keys():
            sf_metrics = valid_results[model_name]['experiments']['slot_filling']['metrics']
            report_lines.append(
                f"| {model_name} | "
                f"{sf_metrics['direct_follow_accuracy']:.1%} | "
                f"{sf_metrics['exact_match_accuracy']:.1%} | "
                f"{sf_metrics['nli_factual_accuracy']:.1%} |"
            )
        
        report_lines.append("\n### Repetition Behavior\n")
        report_lines.append("| Model | Repetition Rate | Avg Cycle Length | Max Cycles |")
        report_lines.append("|-------|-----------------|------------------|------------|")
        
        for model_name in valid_results.keys():
            cd_stats = valid_results[model_name]['experiments']['cycle_detection']['statistics']
            report_lines.append(
                f"| {model_name} | "
                f"{cd_stats['repetition_rate']:.1%} | "
                f"{cd_stats['avg_cycle_length']:.1f} | "
                f"{cd_stats['max_cycle_count']} |"
            )
        
        report_lines.append("\n## Key Findings\n")
        
        # Determine winner for each metric
        sf_winner = max(valid_results.items(), 
                       key=lambda x: x[1]['experiments']['slot_filling']['metrics']['nli_factual_accuracy'])
        
        rep_rate_comparison = {k: v['experiments']['cycle_detection']['statistics']['repetition_rate'] 
                              for k, v in valid_results.items()}
        less_repetitive = min(rep_rate_comparison.items(), key=lambda x: x[1])
        more_repetitive = max(rep_rate_comparison.items(), key=lambda x: x[1])
        
        report_lines.extend([
            f"- **Best Slot-Filling Performance:** {sf_winner[0]} "
            f"({sf_winner[1]['experiments']['slot_filling']['metrics']['nli_factual_accuracy']:.1%} NLI accuracy)",
            f"- **Less Repetitive:** {less_repetitive[0]} ({less_repetitive[1]:.1%} repetition rate)",
            f"- **More Repetitive:** {more_repetitive[0]} ({more_repetitive[1]:.1%} repetition rate)",
        ])
        
        # Architecture comparison
        layer_diff = abs(valid_results[list(valid_results.keys())[0]]['model_info']['num_layers'] - 
                        valid_results[list(valid_results.keys())[1]]['model_info']['num_layers'])
        report_lines.append(f"- **Layer Count Difference:** {layer_diff} layers\n")
        
        report_lines.extend([
            "## Output Files\n",
            f"- **Comparison Plots:** `{self.output_dir / 'comparison_plots'}`",
            f"- **Individual Results:** `{self.output_dir / '[model_name]'}`",
            f"- **This Report:** `{self.output_dir / 'comparison_report.md'}`\n"
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save JSON comparison
        comparison_data = {
            'sample_size': self.sample_size,
            'models': {
                model: {
                    'architecture': results['model_info'],
                    'slot_filling': results['experiments']['slot_filling']['metrics'],
                    'cycles': results['experiments']['cycle_detection']['statistics']
                }
                for model, results in valid_results.items()
            }
        }
        
        json_path = self.output_dir / "comparison_results.json"
        with open(json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(report_text)
        
        print(f"\n✓ Comparison report saved to: {report_path}")
        print(f"✓ JSON comparison saved to: {json_path}")
        
        return report_path
    
    def run_all(self):
        """Run complete comparison pipeline"""
        try:
            self.run_experiments()
            self.create_comparison_plots()
            self.generate_comparison_report()
            
            print("\n" + "=" * 60)
            print("🎉 COMPARISON COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nResults available in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Comparison failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Compare OLMo and Pythia models')
    parser.add_argument('--sample-size', type=int, default=30,
                       help='Number of samples to use (default: 30)')
    parser.add_argument('--output-dir', type=str, default='outputs/olmo_pythia_comparison',
                       help='Output directory')
    parser.add_argument('--olmo-model', type=str, default='allenai/OLMo-1B-hf',
                       help='OLMo model to use')
    parser.add_argument('--pythia-model', type=str, default='EleutherAI/pythia-1.4b',
                       help='Pythia model to use')
    parser.add_argument('--pythia-checkpoint', type=str, default='steplatest',
                       help='Pythia checkpoint to use (default: steplatest)')
    
    args = parser.parse_args()
    
    # Define models to compare
    models = {
        'OLMo-1B': args.olmo_model,
        'Pythia-1.4B': args.pythia_model,
    }
    
    # Create and run comparison
    comparison = ModelComparison(
        models=models,
        output_dir=args.output_dir,
        sample_size=args.sample_size
    )
    
    success = comparison.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
