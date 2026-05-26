#!/usr/bin/env python3
"""
Multi-Layer Causal Attention Intervention Experiment
Tests whether forcing attention heads across multiple layers simultaneously induces repetition.
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

# Import from the original intervention script
from causal_attention_intervention import AttentionInterventionProcessor
from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from modules.cached_data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles

print("✅ All imports successful!")

class MultiLayerInterventionProcessor(AttentionInterventionProcessor):
    """Implements multi-layer attention interventions for causal testing."""
    
    def create_multilayer_attention_hook(self, layer_head_configs, focus_token_id, intervention_strength=0.8):
        """Create hooks that force attention across multiple layers and heads."""
        hooks_dict = {}
        
        for layer_idx, head_list in layer_head_configs.items():
            def make_layer_hook(layer_num, target_heads):
                def multilayer_attention_intervention_hook(module, input, output):
                    # GPT-NeoX returns (hidden_states, attention_weights)
                    if isinstance(output, tuple) and len(output) >= 2:
                        hidden_states, attention_weights = output
                    else:
                        return output
                        
                    if attention_weights is None:
                        return output
                        
                    batch_size, num_heads, seq_len, _ = attention_weights.shape
                    
                    # Apply intervention to specified heads
                    for batch_idx in range(batch_size):
                        for head_idx in target_heads:
                            if head_idx < num_heads:
                                # Method: Force attention to positions likely containing NEWLINE/special tokens
                                for query_pos in range(seq_len):
                                    # Boost attention to last few positions and line breaks
                                    for key_pos in range(max(0, seq_len-8), seq_len):
                                        attention_weights[batch_idx, head_idx, query_pos, key_pos] *= (1 + intervention_strength)
                                    
                                    # Also boost attention to early positions (beginning of sequences)
                                    for key_pos in range(min(5, seq_len)):
                                        attention_weights[batch_idx, head_idx, query_pos, key_pos] *= (1 + intervention_strength * 0.5)
                                
                                # Renormalize attention weights for this head
                                attention_weights[batch_idx, head_idx] = F.softmax(
                                    attention_weights[batch_idx, head_idx], dim=-1
                                )
                    
                    return (hidden_states, attention_weights)
                
                return multilayer_attention_intervention_hook
            
            hooks_dict[layer_idx] = make_layer_hook(layer_idx, head_list)
        
        return hooks_dict
    
    def apply_multilayer_intervention(self, layer_head_configs, focus_token_id, intervention_strength=0.8):
        """Apply multi-layer attention intervention."""
        
        # Get the model layers
        if hasattr(self.model, 'gpt_neox'):  # Pythia/NeoX style
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):  # GPT-style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):  # Other architectures
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        # Create hooks for each layer
        hooks_dict = self.create_multilayer_attention_hook(layer_head_configs, focus_token_id, intervention_strength)
        
        # Register hooks and collect handles
        hook_handles = {}
        for layer_idx, hook_func in hooks_dict.items():
            if layer_idx < len(layers):
                target_layer_module = layers[layer_idx]
                
                # Find attention module
                if hasattr(target_layer_module, 'attention'):  # Pythia/NeoX style
                    attention_module = target_layer_module.attention
                elif hasattr(target_layer_module, 'attn'):
                    attention_module = target_layer_module.attn
                elif hasattr(target_layer_module, 'self_attn'):
                    attention_module = target_layer_module.self_attn
                else:
                    continue  # Skip if can't find attention module
                
                # Register hook
                handle = attention_module.register_forward_hook(hook_func)
                hook_handles[layer_idx] = handle
        
        return hook_handles
    
    def generate_with_multilayer_intervention(self, input_ids, layer_head_configs, focus_token_id, 
                                           max_new_tokens=100, intervention_strength=0.8):
        """Generate text with multi-layer attention intervention."""
        
        # Apply multi-layer intervention
        hook_handles = self.apply_multilayer_intervention(
            layer_head_configs, focus_token_id, intervention_strength
        )
        
        try:
            # Generate with intervention
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            # Always remove all hooks
            for handle in hook_handles.values():
                handle.remove()
        
        return generated
    
    def analyze_multilayer_repetition_induction(self, texts, layer_configs_list, 
                                             intervention_strength=0.8, max_new_tokens=100):
        """Test whether multi-layer attention intervention induces repetition."""
        
        focus_token_id = self.tokenizer.encode('\n')[0] if '\n' in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
        
        results = {
            'baseline_generations': [],
            'intervention_results': {},  # config_key -> generations
            'baseline_cycles': [],
            'intervention_cycles': {},   # config_key -> cycles
            'repetition_induced': {},    # config_key -> [bool]
            'input_texts': texts,
            'layer_configs': layer_configs_list
        }
        
        print(f"🧪 Testing multi-layer attention intervention")
        print(f"💪 Intervention strength: {intervention_strength}")
        print(f"🎯 Focus token: {self.tokenizer.decode([focus_token_id])}")
        print(f"📋 Layer configurations: {len(layer_configs_list)}")
        
        # Initialize results for each layer configuration
        for i, layer_config in enumerate(layer_configs_list):
            config_key = f"config_{i}"
            results['intervention_results'][config_key] = []
            results['intervention_cycles'][config_key] = []
            results['repetition_induced'][config_key] = []
        
        # Process in batches for memory efficiency
        batch_size = 4 if torch.cuda.is_available() else 1
        
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            # Clear GPU cache periodically
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
                
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # 1. Baseline generation (no intervention)
            with torch.no_grad():
                baseline_output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            baseline_text = self.tokenizer.decode(baseline_output[0], skip_special_tokens=False)
            baseline_tokens = self.tokenizer.encode(baseline_text, return_tensors='pt')[0]
            baseline_cycles = detect_cycles(baseline_tokens)
            
            results['baseline_generations'].append(baseline_text)
            results['baseline_cycles'].append(baseline_cycles)
            
            # 2. Multi-layer intervention generations
            for config_idx, layer_config in enumerate(layer_configs_list):
                config_key = f"config_{config_idx}"
                
                intervention_output = self.generate_with_multilayer_intervention(
                    input_ids, layer_config, focus_token_id, 
                    max_new_tokens, intervention_strength
                )
                
                intervention_text = self.tokenizer.decode(intervention_output[0], skip_special_tokens=False)
                intervention_tokens = self.tokenizer.encode(intervention_text, return_tensors='pt')[0]
                intervention_cycles = detect_cycles(intervention_tokens)
                
                # Check if repetition was induced
                baseline_cycle_count = 1 if baseline_cycles is not None else 0
                intervention_cycle_count = 1 if intervention_cycles is not None else 0
                repetition_induced = intervention_cycle_count > baseline_cycle_count
                
                # Store results
                results['intervention_results'][config_key].append(intervention_text)
                results['intervention_cycles'][config_key].append(intervention_cycles)
                results['repetition_induced'][config_key].append(repetition_induced)
                
                if i < 2 and config_idx < 3:  # Print first few examples
                    layers_str = ", ".join([f"L{k}:H{v}" for k, v in layer_config.items()])
                    print(f"\n📝 Example {i+1} - Config {config_idx} ({layers_str}):")
                    print(f"Baseline cycles: {baseline_cycle_count}")
                    print(f"Intervention cycles: {intervention_cycle_count}")
                    print(f"Repetition induced: {repetition_induced}")
        
        return results
    
    def create_multilayer_analysis_plots(self, results, output_dir):
        """Create visualization plots for multi-layer intervention results."""
        
        # Calculate induction rates for each configuration
        config_keys = list(results['repetition_induced'].keys())
        induction_rates = []
        config_labels = []
        layer_counts = []
        
        for i, config_key in enumerate(config_keys):
            rate = np.mean(results['repetition_induced'][config_key])
            induction_rates.append(rate)
            
            # Create readable label
            layer_config = results['layer_configs'][i]
            total_heads = sum(len(heads) for heads in layer_config.values())
            layer_counts.append(total_heads)
            config_labels.append(f"C{i} ({len(layer_config)}L, {total_heads}H)")
        
        # Plot 1: Induction rates by configuration
        plt.figure(figsize=(16, 12))
        
        plt.subplot(3, 2, 1)
        bars = plt.bar(range(len(induction_rates)), induction_rates, color='skyblue', alpha=0.7)
        plt.xlabel('Configuration')
        plt.ylabel('Repetition Induction Rate')
        plt.title('Multi-Layer Intervention Results')
        plt.xticks(range(0, len(config_labels), max(1, len(config_labels)//10)), 
                  config_labels[::max(1, len(config_labels)//10)], rotation=45, ha='right')
        plt.ylim(0, max(1.0, max(induction_rates) * 1.1))
        
        # Plot 2: Number of layers vs effectiveness
        plt.subplot(3, 2, 2)
        num_layers = [len(config) for config in results['layer_configs']]
        plt.scatter(num_layers, induction_rates, alpha=0.7, s=60)
        plt.xlabel('Number of Layers')
        plt.ylabel('Induction Rate')
        plt.title('Layer Count vs Effectiveness')
        
        if len(set(num_layers)) > 1:
            z = np.polyfit(num_layers, induction_rates, 1)
            p = np.poly1d(z)
            plt.plot(sorted(set(num_layers)), p(sorted(set(num_layers))), "r--", alpha=0.8)
        
        # Plot 3: Total heads vs effectiveness
        plt.subplot(3, 2, 3)
        plt.scatter(layer_counts, induction_rates, alpha=0.7, s=60, color='orange')
        plt.xlabel('Total Heads Across Layers')
        plt.ylabel('Induction Rate')
        plt.title('Total Head Count vs Effectiveness')
        
        if len(set(layer_counts)) > 1:
            z = np.polyfit(layer_counts, induction_rates, 1)
            p = np.poly1d(z)
            plt.plot(sorted(set(layer_counts)), p(sorted(set(layer_counts))), "r--", alpha=0.8)
        
        # Plot 4: Distribution of results
        plt.subplot(3, 2, 4)
        plt.hist(induction_rates, bins=max(5, len(set(induction_rates))), alpha=0.7, color='lightgreen')
        plt.xlabel('Induction Rate')
        plt.ylabel('Frequency')
        plt.title('Distribution of Induction Rates')
        plt.axvline(x=np.mean(induction_rates), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(induction_rates):.2%}')
        plt.legend()
        
        # Plot 5: Top configurations
        plt.subplot(3, 2, 5)
        top_indices = np.argsort(induction_rates)[-10:]  # Top 10
        top_rates = [induction_rates[i] for i in top_indices]
        top_labels = [config_labels[i] for i in top_indices]
        
        bars = plt.barh(range(len(top_rates)), top_rates, color='gold', alpha=0.7)
        plt.ylabel('Configuration')
        plt.xlabel('Induction Rate')
        plt.title('Top 10 Configurations')
        plt.yticks(range(len(top_labels)), top_labels)
        
        # Plot 6: Layer distribution analysis
        plt.subplot(3, 2, 6)
        early_layers = []  # layers 0-7
        late_layers = []   # layers 8+
        
        for i, config in enumerate(results['layer_configs']):
            has_early = any(layer < 8 for layer in config.keys())
            has_late = any(layer >= 8 for layer in config.keys())
            
            if has_early and not has_late:
                early_layers.append(induction_rates[i])
            elif has_late and not has_early:
                late_layers.append(induction_rates[i])
        
        if early_layers and late_layers:
            categories = ['Early Layers\n(0-7)', 'Late Layers\n(8+)']
            avg_rates = [np.mean(early_layers), np.mean(late_layers)]
            colors = ['lightcoral', 'lightblue']
            
            bars = plt.bar(categories, avg_rates, color=colors, alpha=0.7)
            plt.ylabel('Average Induction Rate')
            plt.title('Early vs Late Layer Effectiveness')
            
            for bar, rate in zip(bars, avg_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "multilayer_intervention_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Causal Attention Intervention Experiment")
    parser.add_argument("--strength", type=float, default=0.8, help="Intervention strength")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Multi-Layer Causal Attention Intervention Experiment (High Memory)")
    print(f"📋 Parameters:")
    print(f"   - Intervention strength: {args.strength}")
    print(f"   - Number of samples: {args.n_samples}")
    print(f"💾 Memory configuration: Up to 100G available")
    
    # Load model and tokenizer with GPU configuration
    print(f"🤖 Loading model...")
    device = get_device()
    print(f"🔧 Using device: {device}")
    
    # Configure GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔥 GPU detected: {torch.cuda.get_device_name()}")
        print(f"💾 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Print model size and memory usage
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    if torch.cuda.is_available():
        print(f"💾 GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    # Load dataset
    print(f"📚 Loading dataset...")
    try:
        dataset = load_text_dataset("JeanKaddour/minipile")
        test_texts = dataset[:args.n_samples]
    except:
        print("Failed to load JeanKaddour/minipile dataset, using wikitext...")
        dataset = load_text_dataset("wikitext")
        test_texts = dataset[:args.n_samples]
    
    print(f"📊 Loaded {len(test_texts)} non-repeating texts for multi-layer intervention testing")
    
    # Define comprehensive layer-head configurations to test
    layer_configs_list = []
    
    # Single layer configurations (different layers, multiple heads)
    for layer in [5, 10, 15, 19]:
        layer_configs_list.append({layer: [0]})        # Single head
        layer_configs_list.append({layer: [0, 1]})     # Two heads  
        layer_configs_list.append({layer: [0, 1, 2, 3]})  # Four heads
        layer_configs_list.append({layer: list(range(8))})  # Many heads
    
    # Two-layer configurations (early + late)
    for early_layer in [5, 8, 10]:
        for late_layer in [15, 17, 19]:
            layer_configs_list.append({
                early_layer: [0, 1], 
                late_layer: [0, 1]
            })
            layer_configs_list.append({
                early_layer: [0, 1, 2, 3], 
                late_layer: [0, 1, 2, 3]
            })
    
    # Three-layer configurations (early + middle + late)
    layer_configs_list.extend([
        {5: [0, 1], 12: [0, 1], 19: [0, 1]},
        {8: [0, 1, 2], 15: [0, 1, 2], 19: [0, 1, 2]},
        {10: [0, 1, 2, 3], 17: [0, 1, 2, 3], 19: [0, 1, 2, 3]},
    ])
    
    # Multi-layer cascade (progressive intervention)
    layer_configs_list.extend([
        {15: [0], 16: [0], 17: [0], 18: [0], 19: [0]},  # Single head cascade
        {15: [0, 1], 17: [0, 1], 19: [0, 1]},           # Two-head cascade
        {10: [0], 12: [0, 1], 15: [0, 1, 2], 19: [0, 1, 2, 3]},  # Growing cascade
    ])
    
    # Full network interventions (distributed)
    layer_configs_list.extend([
        {i: [0] for i in range(5, 20, 2)},  # Every other layer, head 0
        {i: [0, 1] for i in range(10, 20)}, # Last 10 layers, first 2 heads
        {i: [0, 1, 2, 3] for i in [8, 12, 16, 19]}, # Strategic layers, 4 heads each
    ])
    
    print(f"📋 Testing {len(layer_configs_list)} layer-head configurations")
    
    # Create output directory
    output_dir = Path(f"./plots/multilayer_intervention")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize multi-layer intervention processor
    processor = MultiLayerInterventionProcessor(model, tokenizer)
    
    # Run multi-layer causal intervention experiment
    print(f"🧪 Running multi-layer causal intervention experiment...")
    results = processor.analyze_multilayer_repetition_induction(
        test_texts, layer_configs_list, args.strength
    )
    
    # Save raw results
    results_path = output_dir / f"multilayer_intervention_results.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Create analysis plots
    plot_path = processor.create_multilayer_analysis_plots(results, output_dir)
    print(f"   ✅ Analysis plots saved: {plot_path}")
    
    # Find best configuration
    best_rate = 0
    best_config_idx = 0
    
    for i, config_key in enumerate(results['repetition_induced'].keys()):
        rate = np.mean(results['repetition_induced'][config_key])
        if rate > best_rate:
            best_rate = rate
            best_config_idx = i
    
    best_config = results['layer_configs'][best_config_idx]
    
    # Create detailed report
    report_path = output_dir / f"multilayer_intervention_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Multi-Layer Causal Attention Intervention Report\n\n")
        f.write(f"**Intervention Strength**: {args.strength}  \n")
        f.write(f"**Total Texts Tested**: {len(test_texts)}  \n")
        f.write(f"**Configurations Tested**: {len(layer_configs_list)}  \n\n")
        
        f.write(f"## Best Configuration Results\n\n")
        f.write(f"**Best Configuration**: {best_config}  \n")
        f.write(f"**Best Induction Rate**: {best_rate:.2%}  \n")
        
        # Analyze configuration types
        all_rates = [np.mean(results['repetition_induced'][key]) for key in results['repetition_induced'].keys()]
        
        f.write(f"## Summary Results\n\n")
        f.write(f"- **Overall Average**: {np.mean(all_rates):.2%}\n")
        f.write(f"- **Maximum Rate**: {best_rate:.2%}\n")
        f.write(f"- **Minimum Rate**: {min(all_rates):.2%}\n")
        f.write(f"- **Standard Deviation**: {np.std(all_rates):.2%}\n\n")
        
        if best_rate > 0.20:
            f.write(f"✅ **STRONG HYPOTHESIS SUPPORT**: Multi-layer intervention induces repetition at {best_rate:.1%} rate.\n\n")
            f.write(f"**Implication**: Coordinated attention across layers can effectively trigger repetitive generation.\n\n")
        elif best_rate > 0.10:
            f.write(f"🔶 **MODERATE HYPOTHESIS SUPPORT**: Multi-layer intervention shows {best_rate:.1%} induction rate.\n\n")
            f.write(f"**Implication**: Multi-layer coordination shows promise but needs refinement.\n\n")
        else:
            f.write(f"❌ **HYPOTHESIS NOT SUPPORTED**: Low induction rate ({best_rate:.1%}) across all configurations.\n\n")
            f.write(f"**Implication**: Multi-layer attention manipulation insufficient for reliable repetition induction.\n\n")
        
        # Configuration analysis
        single_layer_rates = []
        multi_layer_rates = []
        
        for i, config in enumerate(results['layer_configs']):
            rate = all_rates[i]
            if len(config) == 1:
                single_layer_rates.append(rate)
            else:
                multi_layer_rates.append(rate)
        
        f.write(f"## Configuration Analysis\n\n")
        if single_layer_rates and multi_layer_rates:
            f.write(f"- **Single Layer Average**: {np.mean(single_layer_rates):.2%}\n")
            f.write(f"- **Multi-Layer Average**: {np.mean(multi_layer_rates):.2%}\n")
            f.write(f"- **Multi-Layer Advantage**: {np.mean(multi_layer_rates) - np.mean(single_layer_rates):+.2%}\n\n")
        
        f.write(f"## Mechanism Insights\n\n")
        f.write(f"- **Layer coordination**: {'Effective' if best_rate > 0.15 else 'Limited'}\n")
        f.write(f"- **Scalability**: {'Good' if len(multi_layer_rates) > 0 and max(multi_layer_rates) > np.mean(single_layer_rates) else 'Poor'}\n")
        f.write(f"- **Network-wide effects**: {'Strong' if best_rate > 0.20 else 'Moderate' if best_rate > 0.10 else 'Weak'}\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\n🎯 Multi-Layer Intervention Summary:")
    print(f"   - Best configuration: {best_config}")
    print(f"   - Best induction rate: {best_rate:.2%}")
    print(f"   - Average across all: {np.mean(all_rates):.2%}")
    print(f"   - Configurations tested: {len(layer_configs_list)}")
    
    if best_rate > 0.15:
        print(f"   ✅ MULTI-LAYER EFFECT: Coordinated cross-layer attention induces repetition")
    else:
        print(f"   ❌ NO MULTI-LAYER EFFECT: Multi-layer intervention ineffective")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()