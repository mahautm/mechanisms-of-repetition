#!/usr/bin/env python3
"""
Aggressive Multi-Strategy Repetition Induction Experiment
Tests multiple strategies progressively until repetition is detected.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
print("✅ Basic imports done")

from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from modules.cached_data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles

print("✅ All imports successful!")

class AggressiveRepetitionInducer:
    """Implements aggressive multi-strategy repetition induction."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        if hasattr(self.model.config, 'num_attention_heads'):
            self.num_heads = self.model.config.num_attention_heads
        elif hasattr(self.model.config, 'n_head'):
            self.num_heads = self.model.config.n_head
        else:
            self.num_heads = 12
        
        # Find special tokens
        self.special_tokens = {}
        for name, token in [('newline', '\n'), ('period', '.'), ('space', ' '), ('comma', ',')]:
            if token in self.tokenizer.get_vocab():
                self.special_tokens[name] = self.tokenizer.encode(token)[0]
            elif name == 'newline' and 'Ċ' in self.tokenizer.get_vocab():
                self.special_tokens[name] = self.tokenizer.encode('Ċ')[0]
        
        if 'newline' not in self.special_tokens:
            self.special_tokens['newline'] = self.tokenizer.eos_token_id
        
        print(f"🎯 Special tokens found: {list(self.special_tokens.keys())}")
    
    def create_aggressive_attention_hook(self, strategy_config):
        """Create hook based on strategy configuration."""
        
        def aggressive_attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, attention_weights = output
            else:
                return output
                
            if attention_weights is None:
                return output
            
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Apply strategy
            for batch_idx in range(batch_size):
                strategy = strategy_config['strategy']
                target_heads = strategy_config['target_heads']
                strength = strategy_config['strength']
                
                if strategy == 'newline_focus':
                    # Focus on newline tokens
                    for head_idx in target_heads:
                        if head_idx < num_heads:
                            # Find positions of target token
                            for pos in range(seq_len):
                                # Boost attention to positions that might be newlines or punctuation
                                if pos > 0:  # Skip first position
                                    attention_weights[batch_idx, head_idx, :, pos] *= (1 + strength)
                            
                            # Renormalize
                            attention_weights[batch_idx, head_idx] = F.softmax(
                                attention_weights[batch_idx, head_idx], dim=-1
                            )
                
                elif strategy == 'last_token_focus':
                    # Focus heavily on the last few tokens
                    for head_idx in target_heads:
                        if head_idx < num_heads:
                            for query_pos in range(seq_len):
                                # Boost attention to last 3 positions
                                for key_pos in range(max(0, seq_len-3), seq_len):
                                    attention_weights[batch_idx, head_idx, query_pos, key_pos] *= (1 + strength)
                            
                            attention_weights[batch_idx, head_idx] = F.softmax(
                                attention_weights[batch_idx, head_idx], dim=-1
                            )
                
                elif strategy == 'recency_bias':
                    # Strong recency bias - focus on recent tokens
                    for head_idx in target_heads:
                        if head_idx < num_heads:
                            for query_pos in range(seq_len):
                                for key_pos in range(seq_len):
                                    # Exponentially increasing weight for more recent positions
                                    recency_mult = 1 + strength * (key_pos / seq_len) ** 2
                                    attention_weights[batch_idx, head_idx, query_pos, key_pos] *= recency_mult
                            
                            attention_weights[batch_idx, head_idx] = F.softmax(
                                attention_weights[batch_idx, head_idx], dim=-1
                            )
                
                elif strategy == 'self_attention':
                    # Force tokens to attend to themselves
                    for head_idx in target_heads:
                        if head_idx < num_heads:
                            for pos in range(seq_len):
                                attention_weights[batch_idx, head_idx, pos, pos] *= (1 + strength)
                            
                            attention_weights[batch_idx, head_idx] = F.softmax(
                                attention_weights[batch_idx, head_idx], dim=-1
                            )
                
                elif strategy == 'uniform_attention':
                    # Make attention uniform (remove structure)
                    for head_idx in target_heads:
                        if head_idx < num_heads:
                            # Set to uniform attention with some noise
                            uniform_attn = torch.ones_like(attention_weights[batch_idx, head_idx]) / seq_len
                            noise = torch.randn_like(uniform_attn) * 0.01
                            attention_weights[batch_idx, head_idx] = F.softmax(uniform_attn + noise, dim=-1)
            
            return (hidden_states, attention_weights)
        
        return aggressive_attention_hook
    
    def apply_aggressive_intervention(self, target_layer, strategy_config):
        """Apply aggressive intervention strategy."""
        
        if hasattr(self.model, 'gpt_neox'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        target_layer_module = layers[target_layer]
        
        if hasattr(target_layer_module, 'attention'):
            attention_module = target_layer_module.attention
        elif hasattr(target_layer_module, 'attn'):
            attention_module = target_layer_module.attn
        elif hasattr(target_layer_module, 'self_attn'):
            attention_module = target_layer_module.self_attn
        else:
            raise ValueError("Cannot find attention module in layer")
        
        hook = self.create_aggressive_attention_hook(strategy_config)
        handle = attention_module.register_forward_hook(hook)
        
        return handle
    
    def generate_with_aggressive_intervention(self, input_ids, target_layer, strategy_config, 
                                           max_new_tokens=200):
        """Generate with aggressive intervention strategy."""
        
        hook_handle = self.apply_aggressive_intervention(target_layer, strategy_config)
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,  # Higher temperature for more variation
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,  # Disable repetition penalty
                )
        finally:
            hook_handle.remove()
        
        return generated
    
    def progressive_aggressive_experiment(self, texts, target_layers=[15, 17, 19], max_new_tokens=200):
        """Run progressive aggressive experiment with multiple strategies."""
        
        # Define strategies to try, ordered from least to most aggressive
        strategies = [
            {'name': 'Newline Focus (2 heads)', 'strategy': 'newline_focus', 'target_heads': [0, 1], 'strength': 1.0},
            {'name': 'Newline Focus (4 heads)', 'strategy': 'newline_focus', 'target_heads': [0, 1, 2, 3], 'strength': 1.5},
            {'name': 'Last Token Focus (2 heads)', 'strategy': 'last_token_focus', 'target_heads': [0, 1], 'strength': 2.0},
            {'name': 'Last Token Focus (4 heads)', 'strategy': 'last_token_focus', 'target_heads': [0, 1, 2, 3], 'strength': 2.0},
            {'name': 'Strong Recency Bias (6 heads)', 'strategy': 'recency_bias', 'target_heads': list(range(6)), 'strength': 2.0},
            {'name': 'Self-Attention Focus (4 heads)', 'strategy': 'self_attention', 'target_heads': [0, 1, 2, 3], 'strength': 3.0},
            {'name': 'All Heads Newline Focus', 'strategy': 'newline_focus', 'target_heads': list(range(self.num_heads)), 'strength': 2.0},
            {'name': 'All Heads Last Token Focus', 'strategy': 'last_token_focus', 'target_heads': list(range(self.num_heads)), 'strength': 3.0},
            {'name': 'Uniform Attention Disruption', 'strategy': 'uniform_attention', 'target_heads': list(range(self.num_heads)), 'strength': 1.0},
        ]
        
        results = {
            'texts': texts,
            'target_layers': target_layers,
            'strategies': strategies,
            'layer_results': {}  # layer -> text_results
        }
        
        print(f"🚀 Running Progressive Aggressive Experiment")
        print(f"📊 Testing {len(strategies)} strategies on {len(target_layers)} layers")
        
        for layer in target_layers:
            print(f"\n🎯 Testing Layer {layer}")
            layer_results = []
            
            for text_idx, text in enumerate(tqdm(texts, desc=f"Layer {layer}")):
                # Tokenize input
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=400)
                input_ids = inputs['input_ids'].to(self.device)
                
                print(f"\n📝 Text {text_idx+1}: {text[:80]}...")
                
                # Baseline
                with torch.no_grad():
                    baseline_output = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.0,
                    )
                
                baseline_text = self.tokenizer.decode(baseline_output[0], skip_special_tokens=False)
                baseline_tokens = self.tokenizer.encode(baseline_text, return_tensors='pt')[0]
                baseline_cycles = detect_cycles(baseline_tokens)
                baseline_has_cycles = baseline_cycles is not None
                
                print(f"   🔄 Baseline: {'CYCLES' if baseline_has_cycles else 'NO CYCLES'}")
                
                text_result = {
                    'input_text': text,
                    'baseline_text': baseline_text,
                    'baseline_cycles': baseline_cycles,
                    'baseline_has_cycles': baseline_has_cycles,
                    'strategy_results': [],
                    'success_strategy': None,
                    'success_found': False
                }
                
                # Try strategies progressively
                for strategy_idx, strategy_config in enumerate(strategies):
                    # Generate with this strategy
                    intervention_output = self.generate_with_aggressive_intervention(
                        input_ids, layer, strategy_config, max_new_tokens
                    )
                    
                    intervention_text = self.tokenizer.decode(intervention_output[0], skip_special_tokens=False)
                    intervention_tokens = self.tokenizer.encode(intervention_text, return_tensors='pt')[0]
                    intervention_cycles = detect_cycles(intervention_tokens)
                    intervention_has_cycles = intervention_cycles is not None
                    
                    # Check if we induced repetition
                    repetition_induced = intervention_has_cycles and not baseline_has_cycles
                    
                    strategy_result = {
                        'strategy_name': strategy_config['name'],
                        'strategy_config': strategy_config,
                        'generated_text': intervention_text,
                        'cycles': intervention_cycles,
                        'has_cycles': intervention_has_cycles,
                        'repetition_induced': repetition_induced
                    }
                    text_result['strategy_results'].append(strategy_result)
                    
                    print(f"      {strategy_config['name']}: {'SUCCESS' if repetition_induced else 'FAIL'}")
                    
                    # If we found success, record it and optionally continue or stop
                    if repetition_induced and not text_result['success_found']:
                        text_result['success_strategy'] = strategy_config['name']
                        text_result['success_found'] = True
                        print(f"      ✅ FIRST SUCCESS: {strategy_config['name']}")
                        # Continue to test remaining strategies for completeness
                
                layer_results.append(text_result)
            
            results['layer_results'][layer] = layer_results
        
        return results
    
    def create_aggressive_analysis_plots(self, results, output_dir):
        """Create comprehensive analysis plots."""
        
        plt.figure(figsize=(20, 16))
        
        # Plot 1: Success rate by strategy
        plt.subplot(3, 4, 1)
        strategy_names = [s['name'] for s in results['strategies']]
        strategy_success_rates = []
        
        for strategy in results['strategies']:
            total_successes = 0
            total_attempts = 0
            
            for layer in results['target_layers']:
                for text_result in results['layer_results'][layer]:
                    for strategy_result in text_result['strategy_results']:
                        if strategy_result['strategy_name'] == strategy['name']:
                            if strategy_result['repetition_induced']:
                                total_successes += 1
                            total_attempts += 1
            
            success_rate = total_successes / total_attempts if total_attempts > 0 else 0
            strategy_success_rates.append(success_rate)
        
        bars = plt.bar(range(len(strategy_names)), strategy_success_rates, alpha=0.7, color='skyblue')
        plt.xlabel('Strategy')
        plt.ylabel('Success Rate')
        plt.title('Strategy Effectiveness')
        plt.xticks(range(len(strategy_names)), [s[:15] + '...' if len(s) > 15 else s for s in strategy_names], rotation=45, ha='right')
        
        # Add value labels
        for bar, rate in zip(bars, strategy_success_rates):
            if rate > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Success rate by layer
        plt.subplot(3, 4, 2)
        layer_success_rates = []
        
        for layer in results['target_layers']:
            total_successes = 0
            total_texts = len(results['layer_results'][layer])
            
            for text_result in results['layer_results'][layer]:
                if text_result['success_found']:
                    total_successes += 1
            
            success_rate = total_successes / total_texts if total_texts > 0 else 0
            layer_success_rates.append(success_rate)
        
        bars = plt.bar([f'L{layer}' for layer in results['target_layers']], layer_success_rates, alpha=0.7, color='lightgreen')
        plt.xlabel('Layer')
        plt.ylabel('Success Rate')
        plt.title('Layer Effectiveness')
        
        for bar, rate in zip(bars, layer_success_rates):
            if rate > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.1%}', ha='center', va='bottom')
        
        # Plot 3: Strategy success heatmap
        plt.subplot(3, 4, 3)
        heatmap_data = np.zeros((len(results['target_layers']), len(strategy_names)))
        
        for layer_idx, layer in enumerate(results['target_layers']):
            for strategy_idx, strategy in enumerate(results['strategies']):
                successes = 0
                attempts = 0
                
                for text_result in results['layer_results'][layer]:
                    for strategy_result in text_result['strategy_results']:
                        if strategy_result['strategy_name'] == strategy['name']:
                            if strategy_result['repetition_induced']:
                                successes += 1
                            attempts += 1
                
                success_rate = successes / attempts if attempts > 0 else 0
                heatmap_data[layer_idx, strategy_idx] = success_rate
        
        sns.heatmap(heatmap_data, 
                   xticklabels=[s[:10] + '...' if len(s) > 10 else s for s in strategy_names],
                   yticklabels=[f'Layer {layer}' for layer in results['target_layers']],
                   annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Strategy × Layer Success Matrix')
        plt.xticks(rotation=45, ha='right')
        
        # Plot 4: First success distribution
        plt.subplot(3, 4, 4)
        first_success_strategies = []
        
        for layer in results['target_layers']:
            for text_result in results['layer_results'][layer]:
                if text_result['success_found']:
                    first_success_strategies.append(text_result['success_strategy'])
        
        if first_success_strategies:
            strategy_counts = {}
            for strategy in first_success_strategies:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            strategies = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            
            plt.pie(counts, labels=[s[:15] + '...' if len(s) > 15 else s for s in strategies], autopct='%1.1f%%')
            plt.title('First Success Strategy Distribution')
        else:
            plt.text(0.5, 0.5, 'No Successes Found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Successful Strategies')
        
        # Plot 5-8: Individual layer details
        for plot_idx, layer in enumerate(results['target_layers']):
            plt.subplot(3, 4, 5 + plot_idx)
            
            layer_data = results['layer_results'][layer]
            text_indices = range(len(layer_data))
            success_mask = [1 if tr['success_found'] else 0 for tr in layer_data]
            
            colors = ['lightgreen' if success else 'lightcoral' for success in success_mask]
            plt.bar(text_indices, success_mask, color=colors, alpha=0.7)
            plt.xlabel('Text Index')
            plt.ylabel('Success (0/1)')
            plt.title(f'Layer {layer} Individual Results')
            plt.ylim(0, 1.2)
        
        # Plot 9: Overall summary
        plt.subplot(3, 4, 9)
        total_texts = sum(len(results['layer_results'][layer]) for layer in results['target_layers'])
        total_successes = sum(sum(1 for tr in results['layer_results'][layer] if tr['success_found']) 
                             for layer in results['target_layers'])
        overall_success_rate = total_successes / total_texts if total_texts > 0 else 0
        
        categories = ['Success', 'Failure']
        values = [total_successes, total_texts - total_successes]
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
        plt.title(f'Overall Success Rate\n{overall_success_rate:.1%} ({total_successes}/{total_texts})')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "aggressive_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Aggressive Multi-Strategy Repetition Induction Experiment")
    parser.add_argument("--layers", nargs='+', type=int, default=[15, 17, 19], help="Target layers")
    parser.add_argument("--n_samples", type=int, default=15, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Aggressive Multi-Strategy Repetition Induction Experiment")
    print(f"📋 Parameters:")
    print(f"   - Target layers: {args.layers}")
    print(f"   - Number of samples: {args.n_samples}")
    
    # Load model and tokenizer
    print(f"🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=args.checkpoint)
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"📚 Loading dataset...")
    try:
        dataset = load_text_dataset("JeanKaddour/minipile")
        test_texts = dataset[:args.n_samples]
    except:
        print("Failed to load JeanKaddour/minipile dataset, using wikitext...")
        dataset = load_text_dataset("wikitext")
        test_texts = dataset[:args.n_samples]
    
    print(f"📊 Testing with {len(test_texts)} texts")
    
    # Create output directory
    output_dir = Path(f"./plots/aggressive_experiment_L{'_'.join(map(str, args.layers))}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize inducer
    inducer = AggressiveRepetitionInducer(model, tokenizer)
    
    # Run aggressive experiment
    print(f"🧪 Running aggressive multi-strategy experiment...")
    results = inducer.progressive_aggressive_experiment(test_texts, args.layers)
    
    # Save raw results
    results_path = output_dir / f"aggressive_results_L{'_'.join(map(str, args.layers))}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Create analysis plots
    plot_path = inducer.create_aggressive_analysis_plots(results, output_dir)
    print(f"   ✅ Analysis plots saved: {plot_path}")
    
    # Analyze results
    total_texts = sum(len(results['layer_results'][layer]) for layer in results['target_layers'])
    total_successes = sum(sum(1 for tr in results['layer_results'][layer] if tr['success_found']) 
                         for layer in results['target_layers'])
    overall_success_rate = total_successes / total_texts if total_texts > 0 else 0
    
    # Create detailed report
    report_path = output_dir / f"aggressive_experiment_report_L{'_'.join(map(str, args.layers))}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Aggressive Multi-Strategy Repetition Induction Report\n\n")
        f.write(f"**Target Layers**: {args.layers}  \n")
        f.write(f"**Total Texts Tested**: {len(test_texts)}  \n")
        f.write(f"**Total Strategy Attempts**: {total_texts}  \n\n")
        
        f.write(f"## Overall Results\n\n")
        f.write(f"- **Overall Success Rate**: {overall_success_rate:.1%} ({total_successes}/{total_texts})\n")
        
        if overall_success_rate >= 0.3:
            f.write(f"\n✅ **STRONG SUCCESS**: Aggressive strategies can reliably induce repetition.\n\n")
        elif overall_success_rate >= 0.1:
            f.write(f"\n⚠️ **MODERATE SUCCESS**: Some aggressive strategies show promise.\n\n")
        else:
            f.write(f"\n❌ **LIMITED SUCCESS**: Aggressive strategies largely ineffective.\n\n")
        
        # Strategy analysis
        f.write(f"## Strategy Effectiveness\n\n")
        
        strategy_stats = {}
        for strategy in results['strategies']:
            total_attempts = 0
            total_successes = 0
            
            for layer in results['target_layers']:
                for text_result in results['layer_results'][layer]:
                    for strategy_result in text_result['strategy_results']:
                        if strategy_result['strategy_name'] == strategy['name']:
                            total_attempts += 1
                            if strategy_result['repetition_induced']:
                                total_successes += 1
            
            success_rate = total_successes / total_attempts if total_attempts > 0 else 0
            strategy_stats[strategy['name']] = {
                'success_rate': success_rate,
                'successes': total_successes,
                'attempts': total_attempts
            }
        
        # Sort by success rate
        sorted_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        for strategy_name, stats in sorted_strategies:
            f.write(f"### {strategy_name}\n")
            f.write(f"- **Success Rate**: {stats['success_rate']:.1%} ({stats['successes']}/{stats['attempts']})\n")
            f.write(f"- **Effectiveness**: {'High' if stats['success_rate'] >= 0.3 else 'Moderate' if stats['success_rate'] >= 0.1 else 'Low'}\n\n")
        
        # Layer analysis
        f.write(f"## Layer Analysis\n\n")
        
        for layer in results['target_layers']:
            layer_successes = sum(1 for tr in results['layer_results'][layer] if tr['success_found'])
            layer_total = len(results['layer_results'][layer])
            layer_success_rate = layer_successes / layer_total if layer_total > 0 else 0
            
            f.write(f"### Layer {layer}\n")
            f.write(f"- **Success Rate**: {layer_success_rate:.1%} ({layer_successes}/{layer_total})\n")
            f.write(f"- **Layer Effectiveness**: {'High' if layer_success_rate >= 0.4 else 'Moderate' if layer_success_rate >= 0.2 else 'Low'}\n\n")
        
        f.write(f"## Key Findings\n\n")
        
        if sorted_strategies:
            best_strategy = sorted_strategies[0]
            f.write(f"- **Most Effective Strategy**: {best_strategy[0]} ({best_strategy[1]['success_rate']:.1%})\n")
        
        best_layer = max(results['target_layers'], 
                        key=lambda l: sum(1 for tr in results['layer_results'][l] if tr['success_found']) / len(results['layer_results'][l]))
        layer_rate = sum(1 for tr in results['layer_results'][best_layer] if tr['success_found']) / len(results['layer_results'][best_layer])
        f.write(f"- **Most Effective Layer**: Layer {best_layer} ({layer_rate:.1%})\n")
        
        f.write(f"- **Repetition Inducible**: {'Yes' if overall_success_rate >= 0.1 else 'Unclear'}\n")
        f.write(f"- **Attention Intervention Effective**: {'Yes' if overall_success_rate >= 0.2 else 'Partially' if overall_success_rate >= 0.05 else 'No'}\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\n🎯 Aggressive Experiment Summary:")
    print(f"   - Overall success rate: {overall_success_rate:.1%} ({total_successes}/{total_texts})")
    
    if sorted_strategies:
        best_strategy = sorted_strategies[0]
        print(f"   - Best strategy: {best_strategy[0]} ({best_strategy[1]['success_rate']:.1%})")
    
    if overall_success_rate >= 0.3:
        print(f"   ✅ STRONG SUCCESS: Aggressive interventions work!")
    elif overall_success_rate >= 0.1:
        print(f"   ⚠️ MODERATE SUCCESS: Some strategies effective")
    else:
        print(f"   ❌ LIMITED SUCCESS: Most strategies ineffective")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()