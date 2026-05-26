#!/usr/bin/env python3
"""
Final Comprehensive Repetition Induction Test
Combines all strategies with robust token detection and simplified logic.
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

class FinalRepetitionTester:
    """Final comprehensive test for repetition induction."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Get model architecture
        if hasattr(self.model.config, 'num_attention_heads'):
            self.num_heads = self.model.config.num_attention_heads
        elif hasattr(self.model.config, 'n_head'):
            self.num_heads = self.model.config.n_head
        else:
            self.num_heads = 12
        
        if hasattr(self.model.config, 'num_hidden_layers'):
            self.num_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            self.num_layers = self.model.config.n_layer
        else:
            self.num_layers = 24
        
        print(f"🏗️ Model: {self.num_layers} layers × {self.num_heads} heads")
        
        # Find focus tokens (punctuation, spaces, etc.)
        self.focus_token_ids = []
        for token in ['.', '!', '?', ',', ';', ':', ' ', '\n']:
            try:
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                if encoded:
                    self.focus_token_ids.extend(encoded)
            except:
                pass
        
        # Remove duplicates and add EOS as fallback
        self.focus_token_ids = list(set(self.focus_token_ids))
        if not self.focus_token_ids:
            self.focus_token_ids = [self.tokenizer.eos_token_id]
        
        print(f"🎯 Focus tokens: {[self.tokenizer.decode([tid]) for tid in self.focus_token_ids[:5]]}")
        
        self.active_hooks = []
    
    def find_focus_positions(self, input_ids):
        """Find positions to focus attention on."""
        positions = []
        for batch_idx, sequence in enumerate(input_ids):
            batch_positions = []
            seq_len = len(sequence)
            
            # Find positions of focus tokens
            for pos, token_id in enumerate(sequence):
                if token_id in self.focus_token_ids:
                    batch_positions.append(pos)
            
            # Always add some strategic positions
            # Last few positions (likely important for generation)
            for pos in range(max(0, seq_len - 3), seq_len):
                if pos not in batch_positions:
                    batch_positions.append(pos)
            
            # Every 10th position for longer sequences
            if seq_len > 20:
                for pos in range(10, seq_len, 10):
                    if pos not in batch_positions:
                        batch_positions.append(pos)
            
            positions.append(sorted(batch_positions))
        return positions
    
    def create_intervention_hook(self, strategy_config, focus_positions):
        """Create intervention hook based on strategy."""
        
        def intervention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, attention_weights = output
            else:
                return output
            
            if attention_weights is None:
                return output
            
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            strategy = strategy_config['strategy']
            target_heads = strategy_config['heads']
            strength = strategy_config['strength']
            
            for batch_idx in range(batch_size):
                if batch_idx < len(focus_positions):
                    positions = focus_positions[batch_idx]
                    
                    for head_idx in target_heads:
                        if head_idx < num_heads:
                            
                            if strategy == 'focus_boost':
                                # Boost attention TO focus positions
                                for pos in positions:
                                    if pos < seq_len:
                                        attention_weights[batch_idx, head_idx, :, pos] *= (1 + strength)
                            
                            elif strategy == 'recency_extreme':
                                # Extreme recency bias
                                for query_pos in range(seq_len):
                                    for key_pos in range(seq_len):
                                        recency_mult = (key_pos / seq_len) ** strength
                                        attention_weights[batch_idx, head_idx, query_pos, key_pos] *= (1 + recency_mult)
                            
                            elif strategy == 'self_attention_boost':
                                # Boost self-attention (diagonal)
                                for pos in range(seq_len):
                                    attention_weights[batch_idx, head_idx, pos, pos] *= (1 + strength)
                            
                            elif strategy == 'uniform_chaos':
                                # Make attention uniform (destroy structure)
                                noise = torch.randn_like(attention_weights[batch_idx, head_idx]) * 0.1
                                attention_weights[batch_idx, head_idx] = F.softmax(
                                    torch.ones_like(attention_weights[batch_idx, head_idx]) + noise, dim=-1
                                )
                                continue  # Skip renormalization
                            
                            elif strategy == 'end_focus':
                                # Focus heavily on last tokens
                                for query_pos in range(seq_len):
                                    for key_pos in range(max(0, seq_len-5), seq_len):
                                        attention_weights[batch_idx, head_idx, query_pos, key_pos] *= (1 + strength)
                            
                            # Renormalize
                            attention_weights[batch_idx, head_idx] = F.softmax(
                                attention_weights[batch_idx, head_idx], dim=-1
                            )
            
            return (hidden_states, attention_weights)
        
        return intervention_hook
    
    def apply_intervention(self, layers, strategy_config, focus_positions):
        """Apply intervention to specified layers."""
        self.clear_hooks()
        
        # Get model layers
        if hasattr(self.model, 'gpt_neox'):
            model_layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            model_layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            model_layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        for layer_idx in layers:
            if layer_idx < len(model_layers):
                layer_module = model_layers[layer_idx]
                
                # Find attention module
                if hasattr(layer_module, 'attention'):
                    attention_module = layer_module.attention
                elif hasattr(layer_module, 'attn'):
                    attention_module = layer_module.attn
                elif hasattr(layer_module, 'self_attn'):
                    attention_module = layer_module.self_attn
                else:
                    continue
                
                hook = self.create_intervention_hook(strategy_config, focus_positions)
                handle = attention_module.register_forward_hook(hook)
                self.active_hooks.append(handle)
    
    def clear_hooks(self):
        """Remove all active hooks."""
        for handle in self.active_hooks:
            handle.remove()
        self.active_hooks = []
    
    def generate_with_intervention(self, input_ids, layers, strategy_config, max_new_tokens=150):
        """Generate with intervention."""
        
        focus_positions = self.find_focus_positions(input_ids)
        self.apply_intervention(layers, strategy_config, focus_positions)
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,  # No repetition penalty
                )
        finally:
            self.clear_hooks()
        
        return generated
    
    def comprehensive_test(self, texts, max_new_tokens=150):
        """Run comprehensive test with all strategies."""
        
        # Define test configurations
        strategies = [
            {'name': 'Single Head Focus', 'strategy': 'focus_boost', 'heads': [0], 'strength': 2.0},
            {'name': 'Multi Head Focus', 'strategy': 'focus_boost', 'heads': [0, 1, 2, 3], 'strength': 3.0},
            {'name': 'All Head Focus', 'strategy': 'focus_boost', 'heads': list(range(self.num_heads)), 'strength': 2.0},
            {'name': 'Extreme Recency', 'strategy': 'recency_extreme', 'heads': [0, 1, 2], 'strength': 3.0},
            {'name': 'Self-Attention Boost', 'strategy': 'self_attention_boost', 'heads': [0, 1, 2, 3], 'strength': 4.0},
            {'name': 'End Token Focus', 'strategy': 'end_focus', 'heads': [0, 1, 2, 3, 4, 5], 'strength': 5.0},
            {'name': 'Attention Chaos', 'strategy': 'uniform_chaos', 'heads': list(range(self.num_heads)), 'strength': 1.0},
        ]
        
        layer_configs = [
            {'name': 'Last Layer Only', 'layers': [self.num_layers - 1]},
            {'name': 'Last 3 Layers', 'layers': list(range(self.num_layers - 3, self.num_layers))},
            {'name': 'All Layers', 'layers': list(range(self.num_layers))},
            {'name': 'Every 4th Layer', 'layers': list(range(0, self.num_layers, 4))},
        ]
        
        results = {
            'texts': texts,
            'strategies': strategies,
            'layer_configs': layer_configs,
            'results_matrix': {},  # (strategy_name, layer_config_name) -> results
            'summary': {}
        }
        
        print(f"🧪 Comprehensive Test: {len(strategies)} strategies × {len(layer_configs)} layer configs")
        
        total_tests = len(texts) * len(strategies) * len(layer_configs)
        successes = 0
        
        with tqdm(total=total_tests, desc="Running comprehensive test") as pbar:
            
            for strategy in strategies:
                for layer_config in layer_configs:
                    test_key = (strategy['name'], layer_config['name'])
                    test_results = []
                    
                    for text_idx, text in enumerate(texts):
                        # Tokenize
                        inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                                             truncation=True, max_length=400)
                        input_ids = inputs['input_ids'].to(self.device)
                        
                        # Baseline
                        with torch.no_grad():
                            baseline_output = self.model.generate(
                                input_ids,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=0.9,
                                top_p=0.95,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                repetition_penalty=1.0,
                            )
                        
                        baseline_text = self.tokenizer.decode(baseline_output[0], skip_special_tokens=False)
                        baseline_tokens = self.tokenizer.encode(baseline_text, return_tensors='pt')[0]
                        baseline_cycles = detect_cycles(baseline_tokens)
                        baseline_has_cycles = baseline_cycles is not None
                        
                        # Intervention
                        intervention_output = self.generate_with_intervention(
                            input_ids, layer_config['layers'], strategy, max_new_tokens
                        )
                        
                        intervention_text = self.tokenizer.decode(intervention_output[0], skip_special_tokens=False)
                        intervention_tokens = self.tokenizer.encode(intervention_text, return_tensors='pt')[0]
                        intervention_cycles = detect_cycles(intervention_tokens)
                        intervention_has_cycles = intervention_cycles is not None
                        
                        # Success if we induced repetition
                        success = intervention_has_cycles and not baseline_has_cycles
                        if success:
                            successes += 1
                        
                        test_results.append({
                            'text_idx': text_idx,
                            'input_text': text,
                            'baseline_cycles': baseline_cycles,
                            'baseline_has_cycles': baseline_has_cycles,
                            'intervention_cycles': intervention_cycles,
                            'intervention_has_cycles': intervention_has_cycles,
                            'success': success
                        })
                        
                        pbar.update(1)
                    
                    results['results_matrix'][test_key] = test_results
        
        # Calculate summary statistics
        print(f"\n🎯 Overall Results:")
        print(f"   - Total tests: {total_tests}")
        print(f"   - Total successes: {successes}")
        print(f"   - Overall success rate: {successes/total_tests:.1%}")
        
        # Find best combinations
        best_success_rate = 0
        best_config = None
        
        for test_key, test_results in results['results_matrix'].items():
            success_count = sum(1 for r in test_results if r['success'])
            success_rate = success_count / len(test_results)
            
            results['summary'][test_key] = {
                'success_count': success_count,
                'total_tests': len(test_results),
                'success_rate': success_rate
            }
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_config = test_key
        
        print(f"   - Best configuration: {best_config}")
        print(f"   - Best success rate: {best_success_rate:.1%}")
        
        return results
    
    def create_comprehensive_plots(self, results, output_dir):
        """Create comprehensive analysis plots."""
        
        # Create summary matrix
        strategies = [s['name'] for s in results['strategies']]
        layer_configs = [lc['name'] for lc in results['layer_configs']]
        
        success_matrix = np.zeros((len(strategies), len(layer_configs)))
        
        for i, strategy_name in enumerate(strategies):
            for j, layer_config_name in enumerate(layer_configs):
                test_key = (strategy_name, layer_config_name)
                if test_key in results['summary']:
                    success_matrix[i, j] = results['summary'][test_key]['success_rate']
        
        plt.figure(figsize=(16, 12))
        
        # Main heatmap
        plt.subplot(2, 3, 1)
        sns.heatmap(success_matrix, 
                   xticklabels=[lc[:15] for lc in layer_configs],
                   yticklabels=[s[:20] for s in strategies],
                   annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('Success Rate Matrix')
        plt.xlabel('Layer Configuration')
        plt.ylabel('Strategy')
        
        # Strategy effectiveness
        plt.subplot(2, 3, 2)
        strategy_means = np.mean(success_matrix, axis=1)
        bars = plt.bar(range(len(strategies)), strategy_means, alpha=0.7, color='skyblue')
        plt.xlabel('Strategy')
        plt.ylabel('Average Success Rate')
        plt.title('Strategy Effectiveness')
        plt.xticks(range(len(strategies)), [s[:15] + '...' if len(s) > 15 else s for s in strategies], 
                  rotation=45, ha='right')
        
        for bar, rate in zip(bars, strategy_means):
            if rate > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Layer configuration effectiveness
        plt.subplot(2, 3, 3)
        layer_means = np.mean(success_matrix, axis=0)
        bars = plt.bar(range(len(layer_configs)), layer_means, alpha=0.7, color='lightgreen')
        plt.xlabel('Layer Configuration')
        plt.ylabel('Average Success Rate')
        plt.title('Layer Configuration Effectiveness')
        plt.xticks(range(len(layer_configs)), [lc[:15] + '...' if len(lc) > 15 else lc for lc in layer_configs], 
                  rotation=45, ha='right')
        
        for bar, rate in zip(bars, layer_means):
            if rate > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Overall distribution
        plt.subplot(2, 3, 4)
        all_rates = [results['summary'][key]['success_rate'] for key in results['summary'].keys()]
        plt.hist(all_rates, bins=10, alpha=0.7, color='orange')
        plt.xlabel('Success Rate')
        plt.ylabel('Frequency')
        plt.title('Success Rate Distribution')
        
        # Top performers
        plt.subplot(2, 3, 5)
        sorted_configs = sorted(results['summary'].items(), 
                               key=lambda x: x[1]['success_rate'], reverse=True)[:5]
        
        top_names = [f"{key[0][:10]}+{key[1][:10]}" for key, _ in sorted_configs]
        top_rates = [value['success_rate'] for _, value in sorted_configs]
        
        bars = plt.bar(range(len(top_names)), top_rates, alpha=0.7, color='gold')
        plt.xlabel('Configuration')
        plt.ylabel('Success Rate')
        plt.title('Top 5 Configurations')
        plt.xticks(range(len(top_names)), top_names, rotation=45, ha='right')
        
        for bar, rate in zip(bars, top_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Summary stats
        plt.subplot(2, 3, 6)
        overall_rate = np.mean(all_rates)
        max_rate = max(all_rates) if all_rates else 0
        
        categories = ['Overall', 'Best Config']
        values = [overall_rate, max_rate]
        colors = ['lightblue', 'gold']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.7)
        plt.ylabel('Success Rate')
        plt.title('Summary Statistics')
        plt.ylim(0, 1)
        
        for bar, rate in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "comprehensive_final_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Final Comprehensive Repetition Induction Test")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Final Comprehensive Repetition Induction Test")
    print(f"📋 Parameters:")
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
    
    # Add some guaranteed texts
    test_texts.extend([
        "The weather is nice today. I think I'll go outside.",
        "Programming can be challenging. But it's also very rewarding.",
        "Books contain vast amounts of knowledge. Reading helps us learn.",
    ])
    
    test_texts = test_texts[:args.n_samples]
    print(f"📊 Testing with {len(test_texts)} texts")
    
    # Create output directory
    output_dir = Path("./plots/final_comprehensive_test")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize tester
    tester = FinalRepetitionTester(model, tokenizer)
    
    # Run comprehensive test
    print(f"🧪 Running final comprehensive test...")
    results = tester.comprehensive_test(test_texts)
    
    # Save results
    results_path = output_dir / "final_comprehensive_results.pt"
    torch.save(results, results_path)
    print(f"   ✅ Results saved: {results_path}")
    
    # Create plots
    plot_path = tester.create_comprehensive_plots(results, output_dir)
    print(f"   ✅ Plots saved: {plot_path}")
    
    # Create report
    report_path = output_dir / "final_comprehensive_report.md"
    
    overall_tests = sum(len(test_results) for test_results in results['results_matrix'].values())
    overall_successes = sum(sum(1 for r in test_results if r['success']) 
                           for test_results in results['results_matrix'].values())
    overall_success_rate = overall_successes / overall_tests if overall_tests > 0 else 0
    
    # Find best configuration
    best_config = max(results['summary'].items(), key=lambda x: x[1]['success_rate'])
    best_name, best_stats = best_config
    
    with open(report_path, 'w') as f:
        f.write(f"# Final Comprehensive Repetition Induction Report\n\n")
        f.write(f"**Total Configurations Tested**: {len(results['results_matrix'])}  \n")
        f.write(f"**Total Individual Tests**: {overall_tests}  \n")
        f.write(f"**Text Samples**: {len(test_texts)}  \n\n")
        
        f.write(f"## Overall Results\n\n")
        f.write(f"- **Overall Success Rate**: {overall_success_rate:.1%} ({overall_successes}/{overall_tests})\n")
        f.write(f"- **Best Configuration**: {best_name[0]} + {best_name[1]}\n")
        f.write(f"- **Best Success Rate**: {best_stats['success_rate']:.1%} ({best_stats['success_count']}/{best_stats['total_tests']})\n\n")
        
        if overall_success_rate >= 0.3:
            f.write(f"✅ **EXCELLENT RESULTS**: Repetition induction is highly achievable with the right configuration.\n\n")
        elif overall_success_rate >= 0.15:
            f.write(f"⚠️ **GOOD RESULTS**: Repetition induction works moderately well with optimized approaches.\n\n")
        elif overall_success_rate >= 0.05:
            f.write(f"⚠️ **LIMITED RESULTS**: Repetition induction shows some promise but needs refinement.\n\n")
        else:
            f.write(f"❌ **MINIMAL RESULTS**: Current approaches largely ineffective for repetition induction.\n\n")
        
        # Top configurations
        f.write(f"## Top Performing Configurations\n\n")
        sorted_configs = sorted(results['summary'].items(), 
                               key=lambda x: x[1]['success_rate'], reverse=True)
        
        for i, (config_name, stats) in enumerate(sorted_configs[:5], 1):
            f.write(f"**#{i}. {config_name[0]} + {config_name[1]}**  \n")
            f.write(f"Success Rate: {stats['success_rate']:.1%} ({stats['success_count']}/{stats['total_tests']})  \n\n")
        
        f.write(f"## Key Insights\n\n")
        f.write(f"- **Intervention Effectiveness**: {'High' if overall_success_rate > 0.2 else 'Moderate' if overall_success_rate > 0.1 else 'Low'}\n")
        f.write(f"- **Configuration Sensitivity**: {'High' if best_stats['success_rate'] > overall_success_rate * 3 else 'Moderate'}\n")
        f.write(f"- **Repeatability**: {'Good' if best_stats['success_count'] >= 2 else 'Limited'}\n")
        f.write(f"- **Practical Viability**: {'Yes' if best_stats['success_rate'] >= 0.3 else 'Limited'}\n")
    
    print(f"   ✅ Report saved: {report_path}")
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   - Overall: {overall_success_rate:.1%} success rate")
    print(f"   - Best config: {best_stats['success_rate']:.1%}")
    print(f"   - Conclusion: {'SUCCESS' if overall_success_rate >= 0.1 else 'LIMITED SUCCESS'}")

if __name__ == "__main__":
    main()