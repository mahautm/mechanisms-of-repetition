#!/usr/bin/env python3
"""
Phase 1 Results Analysis

This script automatically analyzes the results from all three Phase 1 experiments
and provides recommendations for Phase 2 based on the outcomes.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

def load_experiment_results(experiment_dirs: List[str]) -> Dict[str, Any]:
    """Load results from all Phase 1 experiments."""
    results = {}
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        result_files = {
            'gradient_based': 'gradient_based_results.json',
            'embedding_manipulation': 'embedding_manipulation_results.json',
            'residual_interruption': 'residual_interruption_results.json'
        }
        
        for exp_type, filename in result_files.items():
            if exp_type in exp_name:
                filepath = os.path.join(exp_dir, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            results[exp_type] = json.load(f)
                            print(f"✅ Loaded {exp_type} results from {filepath}")
                    except Exception as e:
                        print(f"❌ Error loading {exp_type}: {e}")
                else:
                    print(f"⚠️  Result file not found: {filepath}")
    
    return results

def analyze_experiment_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance across all experiments."""
    analysis = {
        'summary': {},
        'best_approaches': {},
        'performance_comparison': {},
        'recommendations': []
    }
    
    total_success_rate = 0.0
    total_experiments = 0
    best_overall_rate = 0.0
    best_overall_approach = None
    
    # Analyze each experiment type
    for exp_type, data in results.items():
        if 'summary' in data:
            summary = data['summary']
            
            if exp_type == 'gradient_based':
                success_rate = summary.get('success_rate_percent', 0.0)
                total_tests = summary.get('total_texts', 0)
                avg_score = summary.get('average_best_score', 0.0)
                
                analysis['summary'][exp_type] = {
                    'success_rate': success_rate,
                    'total_tests': total_tests,
                    'average_score': avg_score,
                    'approach': 'Gradient Ascent Optimization'
                }
                
            elif exp_type in ['embedding_manipulation', 'residual_interruption']:
                success_rate = summary.get('overall_success_rate', 0.0)
                total_tests = summary.get('total_tests', 0)
                best_strategy = summary.get('best_pattern' if exp_type == 'embedding_manipulation' else 'best_strategy', 'N/A')
                
                analysis['summary'][exp_type] = {
                    'success_rate': success_rate,
                    'total_tests': total_tests,
                    'best_strategy': best_strategy,
                    'approach': 'Embedding Manipulation' if exp_type == 'embedding_manipulation' else 'Residual Interruption'
                }
            
            # Track overall best
            if success_rate > best_overall_rate:
                best_overall_rate = success_rate
                best_overall_approach = exp_type
            
            total_success_rate += success_rate
            total_experiments += 1
    
    # Calculate overall metrics
    avg_success_rate = total_success_rate / total_experiments if total_experiments > 0 else 0.0
    
    analysis['performance_comparison'] = {
        'average_success_rate': avg_success_rate,
        'best_approach': best_overall_approach,
        'best_success_rate': best_overall_rate,
        'improvement_over_baseline': avg_success_rate - 0.0  # Previous experiments were 0.0%
    }
    
    # Generate recommendations
    analysis['recommendations'] = generate_phase2_recommendations(analysis)
    
    return analysis

def generate_phase2_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate Phase 2 recommendations based on Phase 1 results."""
    recommendations = []
    best_rate = analysis['performance_comparison']['best_success_rate']
    avg_rate = analysis['performance_comparison']['average_success_rate']
    
    if best_rate >= 10.0:
        recommendations.extend([
            "🎯 **BREAKTHROUGH ACHIEVED!** Success rate ≥10% detected.",
            f"📈 Focus on optimizing the best approach: {analysis['performance_comparison']['best_approach']}",
            "🔧 Implement parameter sweeping for optimal intervention strength",
            "🔄 Test multi-modal combinations of successful strategies",
            "📊 Scale up to larger test sets (50+ texts) for validation"
        ])
    elif avg_rate >= 5.0:
        recommendations.extend([
            "📈 **PARTIAL SUCCESS** detected. Some approaches showing promise.",
            "🎛️ Increase intervention strengths (2x-5x current values)",
            "🔄 Combine multiple successful strategies simultaneously",
            "⚡ Test extreme parameter ranges for best approaches"
        ])
    elif avg_rate >= 1.0:
        recommendations.extend([
            "⚠️ **MINIMAL SUCCESS** detected. Weak signal observed.",
            "💪 Implement **EXTREME INTERVENTION** strategies:",
            "   - 10x-100x intervention strength scaling",
            "   - Direct logit manipulation at output layer",
            "   - Custom loss functions for repetition rewards",
            "🎯 Focus on most promising approach for extreme optimization"
        ])
    else:
        recommendations.extend([
            "❌ **CONTINUED NULL RESULTS** - Need fundamental approach change.",
            "🚨 **PHASE 3 REQUIRED**: Ultra-aggressive interventions:",
            "   - Training-time interventions with custom objectives",
            "   - Model surgery: direct weight modifications", 
            "   - Architectural modifications (attention head replacement)",
            "   - Fine-tuning with repetition-heavy datasets",
            "🔬 Consider whether repetition induction is fundamentally possible"
        ])
    
    return recommendations

def create_results_visualization(analysis: Dict[str, Any], output_dir: str):
    """Create visualization of Phase 1 results."""
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Success Rate Comparison
    exp_names = []
    success_rates = []
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (exp_type, data) in enumerate(analysis['summary'].items()):
        exp_names.append(data['approach'])
        success_rates.append(data['success_rate'])
    
    bars1 = ax1.bar(exp_names, success_rates, color=colors[:len(exp_names)])
    ax1.set_title('Phase 1: Success Rates by Experiment Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(0, max(max(success_rates) * 1.2, 10))
    
    # Add value labels on bars
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add baseline line
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Previous Baseline (0%)')
    ax1.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Success Threshold (10%)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. Performance vs Baseline
    improvement = [rate for rate in success_rates]
    ax2.bar(exp_names, improvement, color=colors[:len(exp_names)], alpha=0.7)
    ax2.set_title('Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Improvement (percentage points)')
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. Strategy Performance (if available)
    strategy_data = []
    strategy_labels = []
    
    for exp_type, data in analysis['summary'].items():
        if 'best_strategy' in data and data['best_strategy'] != 'N/A':
            strategy_labels.append(f"{exp_type}\n{data['best_strategy']}")
            strategy_data.append(data['success_rate'])
    
    if strategy_data:
        ax3.bar(range(len(strategy_data)), strategy_data, 
               color=colors[:len(strategy_data)])
        ax3.set_title('Best Strategy Performance', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_xticks(range(len(strategy_labels)))
        ax3.set_xticklabels(strategy_labels, rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'No strategy data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Strategy Performance - Data Pending')
    
    # 4. Phase 2 Decision Matrix
    decision_colors = ['red' if max(success_rates) < 1 else 
                      'orange' if max(success_rates) < 5 else
                      'yellow' if max(success_rates) < 10 else 'green']
    
    decision_labels = ['Null\n(<1%)', 'Minimal\n(1-5%)', 'Partial\n(5-10%)', 'Success\n(≥10%)']
    decision_values = [1 if max(success_rates) < 1 else 0,
                      1 if 1 <= max(success_rates) < 5 else 0,
                      1 if 5 <= max(success_rates) < 10 else 0,
                      1 if max(success_rates) >= 10 else 0]
    
    ax4.pie(decision_values, labels=decision_labels, 
           colors=['red', 'orange', 'yellow', 'green'],
           startangle=90, counterclock=False)
    ax4.set_title(f'Phase 1 Outcome\n(Best: {max(success_rates):.1f}%)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'phase1_results_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Results visualization saved: {plot_path}")
    
    return plot_path

def generate_phase2_plan(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed Phase 2 experimental plan."""
    best_rate = analysis['performance_comparison']['best_success_rate']
    best_approach = analysis['performance_comparison']['best_approach']
    
    phase2_plan = {
        'phase': 2,
        'trigger_condition': f"Phase 1 best success rate: {best_rate:.1f}%",
        'recommended_actions': analysis['recommendations'],
        'experiments': [],
        'timeline': '2-3 days',
        'resources': 'Continue 100G GPU allocation'
    }
    
    if best_rate >= 10.0:
        # Parameter optimization phase
        phase2_plan['experiments'] = [
            {
                'name': 'Parameter Sweeping',
                'description': f'Optimize parameters for {best_approach}',
                'parameters': ['intervention_strength: 0.1x to 10x', 'learning_rate: 0.01 to 0.5', 'num_steps: 5 to 50'],
                'priority': 'HIGH'
            },
            {
                'name': 'Multi-Modal Combination',
                'description': 'Combine best strategies from different approaches',
                'priority': 'HIGH'
            },
            {
                'name': 'Scale Validation',
                'description': 'Test on 50+ texts for statistical significance',
                'priority': 'MEDIUM'
            }
        ]
    elif best_rate >= 1.0:
        # Amplification phase
        phase2_plan['experiments'] = [
            {
                'name': 'Extreme Parameter Scaling',
                'description': f'10x-100x intervention strength for {best_approach}',
                'priority': 'HIGH'
            },
            {
                'name': 'Strategy Combination',
                'description': 'Simultaneous multi-strategy application',
                'priority': 'HIGH'
            }
        ]
    else:
        # Fundamental approach change
        phase2_plan['experiments'] = [
            {
                'name': 'Direct Logit Manipulation',
                'description': 'Directly modify output probabilities to force repetition',
                'priority': 'CRITICAL'
            },
            {
                'name': 'Training-Time Interventions',
                'description': 'Custom loss functions with repetition rewards',
                'priority': 'CRITICAL'
            }
        ]
    
    return phase2_plan

def save_comprehensive_report(analysis: Dict[str, Any], phase2_plan: Dict[str, Any], output_dir: str):
    """Save comprehensive analysis report."""
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'phase1_analysis': analysis,
        'phase2_plan': phase2_plan,
        'conclusion': {
            'overall_success': analysis['performance_comparison']['best_success_rate'] >= 10.0,
            'breakthrough_detected': analysis['performance_comparison']['best_success_rate'] >= 5.0,
            'next_phase_required': True,
            'confidence_level': 'High' if analysis['performance_comparison']['best_success_rate'] >= 10.0 else
                               'Medium' if analysis['performance_comparison']['best_success_rate'] >= 1.0 else 'Low'
        }
    }
    
    # Save JSON report
    report_path = os.path.join(output_dir, 'phase1_comprehensive_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save markdown summary
    md_path = os.path.join(output_dir, 'phase1_analysis_summary.md')
    with open(md_path, 'w') as f:
        f.write(f"# Phase 1 Aggressive Experiments - Analysis Report\n\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 🎯 Executive Summary\n\n")
        best_rate = analysis['performance_comparison']['best_success_rate']
        f.write(f"- **Best Success Rate**: {best_rate:.1f}%\n")
        f.write(f"- **Best Approach**: {analysis['performance_comparison']['best_approach']}\n")
        f.write(f"- **Average Success Rate**: {analysis['performance_comparison']['average_success_rate']:.1f}%\n")
        f.write(f"- **Improvement over Baseline**: +{analysis['performance_comparison']['improvement_over_baseline']:.1f} percentage points\n\n")
        
        f.write("## 📊 Experiment Results\n\n")
        for exp_type, data in analysis['summary'].items():
            f.write(f"### {data['approach']}\n")
            f.write(f"- Success Rate: {data['success_rate']:.1f}%\n")
            f.write(f"- Total Tests: {data.get('total_tests', 'N/A')}\n")
            if 'best_strategy' in data:
                f.write(f"- Best Strategy: {data['best_strategy']}\n")
            f.write("\n")
        
        f.write("## 🚀 Phase 2 Recommendations\n\n")
        for i, rec in enumerate(analysis['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
        f.write("\n")
        
        f.write("## 📋 Phase 2 Experimental Plan\n\n")
        for exp in phase2_plan['experiments']:
            f.write(f"### {exp['name']} ({exp['priority']} Priority)\n")
            f.write(f"{exp['description']}\n\n")
    
    print(f"📝 Comprehensive report saved: {report_path}")
    print(f"📝 Summary report saved: {md_path}")
    
    return report_path, md_path

def main():
    """Main analysis function."""
    print("🔍 Phase 1 Results Analysis Starting...")
    print("=" * 60)
    
    # Define experiment directories
    base_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots"
    experiment_dirs = [
        os.path.join(base_dir, "gradient_based_experiment"),
        os.path.join(base_dir, "embedding_manipulation_experiment"),
        os.path.join(base_dir, "residual_interruption_experiment")
    ]
    
    # Load results
    results = load_experiment_results(experiment_dirs)
    
    if not results:
        print("❌ No experiment results found. Experiments may still be running.")
        print("   Check back when Phase 1 experiments are complete.")
        return
    
    # Analyze performance
    print(f"\n📊 Analyzing {len(results)} completed experiments...")
    analysis = analyze_experiment_performance(results)
    
    # Generate Phase 2 plan
    phase2_plan = generate_phase2_plan(analysis)
    
    # Create output directory
    output_dir = os.path.join(base_dir, "phase1_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    plot_path = create_results_visualization(analysis, output_dir)
    
    # Save comprehensive report
    report_path, md_path = save_comprehensive_report(analysis, phase2_plan, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 PHASE 1 ANALYSIS COMPLETE")
    print("=" * 60)
    
    best_rate = analysis['performance_comparison']['best_success_rate']
    print(f"📈 Best Success Rate: {best_rate:.1f}%")
    print(f"🏆 Best Approach: {analysis['performance_comparison']['best_approach']}")
    print(f"📊 Average Success Rate: {analysis['performance_comparison']['average_success_rate']:.1f}%")
    
    print(f"\n🎯 PHASE 2 RECOMMENDATION:")
    if best_rate >= 10.0:
        print("✅ BREAKTHROUGH ACHIEVED! Proceed with parameter optimization.")
    elif best_rate >= 5.0:
        print("📈 PARTIAL SUCCESS! Amplify successful approaches.")  
    elif best_rate >= 1.0:
        print("⚠️ MINIMAL SUCCESS! Implement extreme interventions.")
    else:
        print("❌ NULL RESULTS CONTINUE! Fundamental approach change needed.")
    
    print(f"\n📂 Results saved in: {output_dir}")
    
    return analysis, phase2_plan

if __name__ == "__main__":
    main()