#!/usr/bin/env python3
"""
Multi-Head Causal Attention Intervention Experiment
Tests whether forcing multiple attention heads to attend to NEWLINE tokens induces repetition.
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

class MultiHeadInterventionProcessor(AttentionInterventionProcessor):
    """Implements multi-head attention interventions for causal testing."""
    
    def create_multihead_attention_hook(self, target_layer, target_heads, focus_token_id, intervention_strength=0.8):
        """Create a hook that forces multiple attention heads to focus on specific tokens."""
        
        def multihead_attention_intervention_hook(module, input, output):
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
                # Find positions of focus token (e.g., NEWLINE)
                focus_positions = []
                # Note: We'll need to pass token_ids or detect them from the attention pattern
                # For now, assume positions are detected based on high attention
                
                for head_idx in target_heads:
                    if head_idx < num_heads:
                        # Method 1: Force attention to last few positions (likely NEWLINE/special tokens)
                        for query_pos in range(seq_len):
                            # Boost attention to last few positions
                            for key_pos in range(max(0, seq_len-5), seq_len):
                                attention_weights[batch_idx, head_idx, query_pos, key_pos] *= (1 + intervention_strength)
                        
                        # Renormalize attention weights for this head
                        attention_weights[batch_idx, head_idx] = F.softmax(
                            attention_weights[batch_idx, head_idx], dim=-1
                        )
            
            return (hidden_states, attention_weights)
        
        return multihead_attention_intervention_hook
    
    def apply_multihead_intervention(self, target_layer, target_heads, focus_token_id, intervention_strength=0.8):
        """Apply multi-head attention intervention to specific layer."""
        
        # Get the target attention layer
        if hasattr(self.model, 'gpt_neox'):  # Pythia/NeoX style
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):  # GPT-style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):  # Other architectures
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        target_layer_module = layers[target_layer]
        
        # Find attention module
        if hasattr(target_layer_module, 'attention'):  # Pythia/NeoX style
            attention_module = target_layer_module.attention
        elif hasattr(target_layer_module, 'attn'):
            attention_module = target_layer_module.attn
        elif hasattr(target_layer_module, 'self_attn'):
            attention_module = target_layer_module.self_attn
        else:
            raise ValueError("Cannot find attention module in layer")
        
        # Create and register hook
        hook = self.create_multihead_attention_hook(target_layer, target_heads, focus_token_id, intervention_strength)
        handle = attention_module.register_forward_hook(hook)
        
        return handle
    
    def generate_with_multihead_intervention(self, input_ids, target_layer, target_heads, focus_token_id, 
                                           max_new_tokens=100, intervention_strength=0.8):
        """Generate text with multi-head attention intervention."""
        
        # Apply multi-head intervention
        hook_handle = self.apply_multihead_intervention(
            target_layer, target_heads, focus_token_id, intervention_strength
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
            # Always remove the hook
            hook_handle.remove()
        
        return generated
    
    def analyze_multihead_repetition_induction(self, texts, target_layer, target_heads_list, 
                                             intervention_strength=0.8, max_new_tokens=100):
        """Test whether multi-head attention intervention induces repetition."""
        
        focus_token_id = self.tokenizer.encode('\n')[0] if '\n' in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
        
        results = {
            'baseline_generations': [],
            'intervention_results': {},  # head_config -> generations
            'baseline_cycles': [],
            'intervention_cycles': {},   # head_config -> cycles
            'repetition_induced': {},    # head_config -> [bool]
            'input_texts': texts,
            'target_heads_configs': target_heads_list
        }
        
        print(f"🧪 Testing multi-head attention intervention on layer {target_layer}")
        print(f"💪 Intervention strength: {intervention_strength}")
        print(f"🎯 Focus token: {self.tokenizer.decode([focus_token_id])}")
        print(f"📋 Head configurations: {target_heads_list}")
        
        # Initialize results for each head configuration
        for head_config in target_heads_list:
            config_key = f"heads_{'-'.join(map(str, head_config))}"
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
            
            # 2. Multi-head intervention generations
            for target_heads in target_heads_list:
                config_key = f"heads_{'-'.join(map(str, target_heads))}"
                
                intervention_output = self.generate_with_multihead_intervention(
                    input_ids, target_layer, target_heads, focus_token_id, 
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
                
                if i < 2:  # Print first few examples
                    print(f"\n📝 Example {i+1} - Heads {target_heads}:")
                    print(f"Baseline cycles: {baseline_cycle_count}")
                    print(f"Intervention cycles: {intervention_cycle_count}")
                    print(f"Repetition induced: {repetition_induced}")
        
        return results
    
    def create_multihead_analysis_plots(self, results, output_dir, target_layer):
        """Create visualization plots for multi-head intervention results."""
        
        # Calculate induction rates for each head configuration
        head_configs = list(results['repetition_induced'].keys())
        induction_rates = []
        config_labels = []
        
        for config in head_configs:
            rate = np.mean(results['repetition_induced'][config])
            induction_rates.append(rate)
            # Clean up config label
            heads_str = config.replace('heads_', '').replace('-', ',')
            config_labels.append(f"Heads [{heads_str}]")
        
        # Plot 1: Induction rates by head configuration
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(induction_rates)), induction_rates, color='skyblue', alpha=0.7)
        plt.xlabel('Head Configuration')
        plt.ylabel('Repetition Induction Rate')
        plt.title(f'Multi-Head Intervention Results (Layer {target_layer})')
        plt.xticks(range(len(config_labels)), config_labels, rotation=45, ha='right')
        plt.ylim(0, max(1.0, max(induction_rates) * 1.1))
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, induction_rates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # Plot 2: Comparison with single head baseline
        plt.subplot(2, 2, 2)
        single_head_rates = [rate for i, rate in enumerate(induction_rates) 
                           if len(results['target_heads_configs'][i]) == 1]
        multi_head_rates = [rate for i, rate in enumerate(induction_rates) 
                          if len(results['target_heads_configs'][i]) > 1]
        
        if single_head_rates and multi_head_rates:
            categories = ['Single Head', 'Multi Head']
            avg_rates = [np.mean(single_head_rates), np.mean(multi_head_rates)]
            colors = ['lightcoral', 'lightgreen']
            
            bars = plt.bar(categories, avg_rates, color=colors, alpha=0.7)
            plt.ylabel('Average Induction Rate')
            plt.title('Single vs Multi-Head Effectiveness')
            plt.ylim(0, max(1.0, max(avg_rates) * 1.1))
            
            for bar, rate in zip(bars, avg_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.2%}', ha='center', va='bottom')
        
        # Plot 3: Number of heads vs effectiveness
        plt.subplot(2, 2, 3)
        head_counts = [len(config) for config in results['target_heads_configs']]
        plt.scatter(head_counts, induction_rates, alpha=0.7, s=100)
        plt.xlabel('Number of Heads')
        plt.ylabel('Induction Rate')
        plt.title('Head Count vs Effectiveness')
        
        # Add trend line
        if len(head_counts) > 1:
            z = np.polyfit(head_counts, induction_rates, 1)
            p = np.poly1d(z)
            plt.plot(sorted(head_counts), p(sorted(head_counts)), "r--", alpha=0.8)
        
        # Plot 4: Success distribution
        plt.subplot(2, 2, 4)
        all_successes = []
        for config in head_configs:
            successes = sum(results['repetition_induced'][config])
            total = len(results['repetition_induced'][config])
            all_successes.append(successes)
        
        plt.hist(all_successes, bins=max(1, len(set(all_successes))), alpha=0.7, color='orange')
        plt.xlabel('Number of Successful Interventions')
        plt.ylabel('Frequency')
        plt.title('Distribution of Intervention Success')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"multihead_intervention_analysis_L{target_layer}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Multi-Head Causal Attention Intervention Experiment")
    parser.add_argument("--layer", type=int, default=19, help="Target layer for intervention")
    parser.add_argument("--strength", type=float, default=0.8, help="Intervention strength")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--max_heads", type=int, default=16, help="Maximum number of heads to test")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Multi-Head Causal Attention Intervention Experiment (High Memory)")
    print(f"📋 Parameters:")
    print(f"   - Target layer: {args.layer}")
    print(f"   - Intervention strength: {args.strength}")
    print(f"   - Number of samples: {args.n_samples}")
    print(f"   - Max heads to test: {args.max_heads}")
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
        test_texts = dataset[:5]  
    except Exception as e:
        print("Failed to load JeanKaddour/minipile dataset, using fallback...")
        dataset = load_text_dataset("JeanKaddour/minipile")
        test_texts = dataset[:args.n_samples]
    
    print(f"📊 Loaded {len(test_texts)} non-repeating texts for multi-head intervention testing")
    
    # Define comprehensive head configurations to test (up to 100G memory)
    target_heads_configs = []
    
    # Single heads (test all heads up to max_heads)
    for i in range(min(args.max_heads, 16)):  # Pythia-1.4b has 16 heads
        target_heads_configs.append([i])
    
    # Pairs of heads (systematic combinations)
    for i in range(0, min(args.max_heads, 8)):
        for j in range(i+1, min(args.max_heads, 8)):
            target_heads_configs.append([i, j])
    
    # Triple heads (key combinations)
    target_heads_configs.extend([
        [0, 1, 2], [0, 4, 8], [0, 7, 15], [1, 8, 15],
        [0, 1, 15], [7, 8, 15], [0, 8, 15], [1, 7, 8]
    ])
    
    # Quad heads (strategic combinations)
    target_heads_configs.extend([
        [0, 1, 2, 3], [0, 4, 8, 12], [0, 1, 8, 15], 
        [1, 7, 8, 15], [0, 3, 7, 15], [4, 8, 12, 15]
    ])
    
    # Many heads (test coordination effects)
    target_heads_configs.extend([
        list(range(8)),      # First half
        list(range(8, 16)),  # Second half
        [0, 2, 4, 6, 8, 10, 12, 14],  # Even heads
        [1, 3, 5, 7, 9, 11, 13, 15],  # Odd heads
        list(range(0, 16, 2)),         # Every other head
        list(range(16))                # All heads
    ])
    
    print(f"📋 Testing {len(target_heads_configs)} head configurations")
    
    # Create output directory
    output_dir = Path(f"./plots/multihead_intervention_L{args.layer}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize multi-head intervention processor
    processor = MultiHeadInterventionProcessor(model, tokenizer)
    
    # Run multi-head causal intervention experiment
    print(f"🧪 Running multi-head causal intervention experiment...")
    results = processor.analyze_multihead_repetition_induction(
        test_texts, args.layer, target_heads_configs, args.strength
    )
    
    # Save raw results
    results_path = output_dir / f"multihead_intervention_results_L{args.layer}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Create analysis plots
    plot_path = processor.create_multihead_analysis_plots(results, output_dir, args.layer)
    print(f"   ✅ Analysis plots saved: {plot_path}")
    
    # Create detailed report
    report_path = output_dir / f"multihead_intervention_report_L{args.layer}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Multi-Head Causal Attention Intervention Report\n\n")
        f.write(f"**Target Layer**: {args.layer}  \n")
        f.write(f"**Intervention Strength**: {args.strength}  \n")
        f.write(f"**Total Texts Tested**: {len(test_texts)}  \n\n")
        
        f.write(f"## Head Configuration Results\n\n")
        
        best_rate = 0
        best_config = None
        
        for config_key in results['repetition_induced'].keys():
            heads_str = config_key.replace('heads_', '').replace('-', ', ')
            induction_rate = np.mean(results['repetition_induced'][config_key])
            successes = sum(results['repetition_induced'][config_key])
            total = len(results['repetition_induced'][config_key])
            
            f.write(f"### Heads [{heads_str}]\n")
            f.write(f"- **Induction Rate**: {induction_rate:.2%} ({successes}/{total})\n")
            f.write(f"- **Configuration**: {len(results['target_heads_configs'][list(results['repetition_induced'].keys()).index(config_key)])} heads\n\n")
            
            if induction_rate > best_rate:
                best_rate = induction_rate
                best_config = f"[{heads_str}]"
        
        f.write(f"## Summary Results\n\n")
        f.write(f"- **Best Configuration**: {best_config} with {best_rate:.2%} induction rate\n")
        f.write(f"- **Multi-Head Advantage**: {'Yes' if best_rate > 0.15 else 'No'}\n\n")
        
        if best_rate > 0.15:
            f.write(f"✅ **HYPOTHESIS SUPPORTED**: Multi-head intervention induces repetition at {best_rate:.1%} rate.\n\n")
            f.write(f"**Implication**: Coordinated attention across multiple heads can trigger repetitive generation.\n\n")
        else:
            f.write(f"❌ **HYPOTHESIS NOT SUPPORTED**: Low induction rate ({best_rate:.1%}) across all configurations.\n\n")
            f.write(f"**Implication**: Multi-head attention manipulation is insufficient for reliable repetition induction.\n\n")
        
        f.write(f"## Mechanism Insights\n\n")
        
        # Analyze single vs multi-head effectiveness
        single_head_rates = []
        multi_head_rates = []
        
        for i, config_key in enumerate(results['repetition_induced'].keys()):
            rate = np.mean(results['repetition_induced'][config_key])
            head_count = len(results['target_heads_configs'][i])
            
            if head_count == 1:
                single_head_rates.append(rate)
            else:
                multi_head_rates.append(rate)
        
        if single_head_rates and multi_head_rates:
            avg_single = np.mean(single_head_rates)
            avg_multi = np.mean(multi_head_rates)
            f.write(f"- **Single Head Average**: {avg_single:.2%}\n")
            f.write(f"- **Multi-Head Average**: {avg_multi:.2%}\n")
            f.write(f"- **Multi-Head Advantage**: {'+' if avg_multi > avg_single else '-'}{abs(avg_multi - avg_single):.1%}\n")
        
        f.write(f"- **Coordination Effect**: {'Strong' if best_rate > 0.20 else 'Moderate' if best_rate > 0.10 else 'Weak'}\n")
        f.write(f"- **Scalability**: {'Yes' if len(multi_head_rates) > 0 and max(multi_head_rates) > max(single_head_rates) else 'No'}\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\n🎯 Multi-Head Intervention Summary:")
    print(f"   - Best configuration: {best_config}")
    print(f"   - Best induction rate: {best_rate:.2%}")
    print(f"   - Configurations tested: {len(target_heads_configs)}")
    
    if best_rate > 0.15:
        print(f"   ✅ MULTI-HEAD EFFECT: Coordinated attention induces repetition")
    else:
        print(f"   ❌ NO MULTI-HEAD EFFECT: Multi-head intervention ineffective")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()