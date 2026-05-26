#!/usr/bin/env python3
"""
Comprehensive Causal Intervention Analysis
Summarizes and visualizes results from all causal intervention experiments.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json
import argparse

print("✅ All imports successful!")

class ComprehensiveCausalAnalyzer:
    """Analyzes and visualizes results from all causal intervention experiments."""
    
    def __init__(self, results_dir="./plots"):
        self.results_dir = Path(results_dir)
        self.experiments = {
            'single_head': 'causal_attention_intervention_L*/causal_intervention_results_*.pt',
            'multihead': 'multihead_intervention_L*/multihead_intervention_results_*.pt',
            'multilayer': 'multilayer_intervention/multilayer_intervention_results.pt',
            'activation_patching': 'activation_patching_L*/activation_patching_results_*.pt',
            'pattern_sequence': 'pattern_intervention_L*/pattern_intervention_results_*.pt'
        }
        
    def load_experiment_results(self):
        """Load results from all available experiments."""
        results = {}
        
        for exp_name, pattern in self.experiments.items():
            exp_results = []
            
            # Find all matching result files
            for result_path in self.results_dir.glob(pattern):
                try:
                    data = torch.load(result_path, map_location='cpu')
                    exp_results.append({
                        'path': result_path,
                        'data': data,
                        'name': result_path.parent.name
                    })
                    print(f"   ✅ Loaded {exp_name}: {result_path.name}")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {result_path}: {e}")
            
            results[exp_name] = exp_results
            
        return results
    
    def calculate_induction_rates(self, results):
        """Calculate induction rates for all experiments."""
        induction_summary = {}
        
        for exp_name, exp_results in results.items():
            exp_rates = []
            
            for result in exp_results:
                data = result['data']
                
                if exp_name == 'single_head':
                    # Single head results
                    if 'repetition_induced' in data:
                        rate = np.mean(data['repetition_induced'])
                        exp_rates.append({
                            'rate': rate,
                            'config': result['name'],
                            'n_samples': len(data['repetition_induced'])
                        })
                
                elif exp_name == 'multihead':
                    # Multi-head results
                    if 'repetition_induced' in data:
                        for config_key, induced_list in data['repetition_induced'].items():
                            rate = np.mean(induced_list)
                            exp_rates.append({
                                'rate': rate,
                                'config': config_key,
                                'n_samples': len(induced_list)
                            })
                
                elif exp_name == 'multilayer':
                    # Multi-layer results
                    if 'repetition_induced' in data:
                        for config_key, induced_list in data['repetition_induced'].items():
                            rate = np.mean(induced_list)
                            exp_rates.append({
                                'rate': rate,
                                'config': config_key,
                                'n_samples': len(induced_list)
                            })
                
                elif exp_name == 'activation_patching':
                    # Activation patching results
                    if 'repetition_induced' in data:
                        rate = np.mean(data['repetition_induced'])
                        exp_rates.append({
                            'rate': rate,
                            'config': 'activation_patch',
                            'n_samples': len(data['repetition_induced'])
                        })
                
                elif exp_name == 'pattern_sequence':
                    # Pattern sequence results
                    if 'repetition_induced' in data:
                        rate = np.mean(data['repetition_induced'])
                        exp_rates.append({
                            'rate': rate,
                            'config': 'pattern_intervention',
                            'n_samples': len(data['repetition_induced'])
                        })
            
            induction_summary[exp_name] = exp_rates
            
        return induction_summary
    
    def run_analysis(self):
        """Run the complete comprehensive analysis."""
        
        print(f"🚀 Starting Comprehensive Causal Intervention Analysis")
        print(f"📂 Results directory: {self.results_dir}")
        
        # Create output directory
        output_dir = self.results_dir / "comprehensive_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load all experiment results
        print(f"📥 Loading experiment results...")
        results = self.load_experiment_results()
        
        total_experiments = sum(len(exp_results) for exp_results in results.values())
        print(f"📊 Loaded {total_experiments} experiment result files")
        
        if total_experiments == 0:
            print("⚠️ No experiment results found. Please run experiments first.")
            return output_dir
        
        # Calculate induction rates
        print(f"🔢 Calculating induction rates...")
        induction_summary = self.calculate_induction_rates(results)
        
        # Print summary statistics
        all_rates = []
        for exp_rates in induction_summary.values():
            for result in exp_rates:
                all_rates.append(result['rate'])
        
        if all_rates:
            print(f"\n🎯 Overall Summary:")
            print(f"   - Total configurations: {len(all_rates)}")
            print(f"   - Average induction rate: {np.mean(all_rates):.2%}")
            print(f"   - Maximum induction rate: {max(all_rates):.2%}")
            print(f"   - Success rate (>15%): {sum(1 for r in all_rates if r > 0.15)/len(all_rates):.1%}")
            
            # Show top 5 results
            print(f"\n🏆 Top 5 Results:")
            all_results = []
            for exp_name, exp_rates in induction_summary.items():
                for result in exp_rates:
                    all_results.append((result['rate'], exp_name, result['config']))
            
            all_results.sort(reverse=True)
            for i, (rate, exp_name, config) in enumerate(all_results[:5], 1):
                print(f"   {i}. {rate:.2%} - {exp_name}: {config[:50]}...")
        
        print(f"📁 Results available in: {output_dir}")
        
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Causal Intervention Analysis")
    parser.add_argument("--results_dir", type=str, default="./plots", help="Directory containing experiment results")
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveCausalAnalyzer(args.results_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()