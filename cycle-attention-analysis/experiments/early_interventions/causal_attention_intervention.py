#!/usr/bin/env python3
"""
Causal Attention Intervention Experiment
Tests whether forcing attention to NEWLINE tokens can induce repetition in non-repeating sentences.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import torch.nn.functional as F
print("✅ torch imported")
from tqdm import tqdm
import time
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
print("✅ Basic imports done")

try:
    from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
    print("✅ model_utils imported")
except ImportError as e:
    print(f"❌ Failed to import model_utils: {e}")
    raise

try:
    from modules.cached_data_utils import load_text_dataset
    print("✅ cached_data_utils imported")
except ImportError as e:
    print(f"❌ Failed to import cached_data_utils: {e}")
    raise

try:
    from parrots.cycle_detection import detect_cycles
    print("✅ cycle_detection imported")
except ImportError as e:
    print(f"❌ Failed to import cycle_detection: {e}")
    raise

print("✅ All imports successful!")

class AttentionInterventionProcessor:
    """Processes sequences with controlled attention interventions."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Store original forward methods for restoration
        self.original_forward_methods = {}
        
    def find_newline_positions(self, input_ids):
        """Find positions of NEWLINE tokens in the sequence."""
        # Common newline token IDs (model-specific)
        newline_candidates = [
            self.tokenizer.encode('\n', add_special_tokens=False),
            self.tokenizer.encode('Ċ', add_special_tokens=False),  # GPT-style
            self.tokenizer.encode('čĊ', add_special_tokens=False),
        ]
        
        newline_token_ids = set()
        for candidates in newline_candidates:
            newline_token_ids.update(candidates)
        
        # Find all newline positions
        newline_positions = []
        for batch_idx, sequence in enumerate(input_ids):
            positions = []
            for pos_idx, token_id in enumerate(sequence):
                if token_id.item() in newline_token_ids:
                    positions.append(pos_idx)
            newline_positions.append(positions)
        
        return newline_positions
    
    def create_attention_hook(self, target_layer, target_head, newline_positions, intervention_strength=0.8):
        """Create a hook that forces attention to NEWLINE tokens."""
        
        def attention_intervention_hook(module, input, output):
            # GPT-NeoX returns (hidden_states, attention_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, attention_weights = output
            else:
                # If output format is unexpected, print for debugging
                print(f"🔍 Unexpected attention output format: {type(output)}, shape: {getattr(output, 'shape', 'no shape')}")
                if isinstance(output, tuple):
                    print(f"   Tuple length: {len(output)}")
                    for i, item in enumerate(output):
                        print(f"   Item {i}: {type(item)}, shape: {getattr(item, 'shape', 'no shape')}")
                return output
                
            if attention_weights is None:
                return output
                
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Only modify the target head
            if target_head < num_heads:
                for batch_idx in range(batch_size):
                    if batch_idx < len(newline_positions) and newline_positions[batch_idx]:
                        # Get newline positions for this batch
                        nl_positions = newline_positions[batch_idx]
                        
                        # Create intervention: increase attention to newline positions
                        for nl_pos in nl_positions:
                            if nl_pos < seq_len:
                                # Increase attention from all positions to this newline position
                                attention_weights[batch_idx, target_head, :, nl_pos] *= (1 + intervention_strength)
                        
                        # Renormalize attention weights for this head
                        attention_weights[batch_idx, target_head] = F.softmax(
                            attention_weights[batch_idx, target_head], dim=-1
                        )
            
            # Return the modified tuple
            if isinstance(output, tuple) and len(output) >= 2:
                return (hidden_states, attention_weights)
            else:
                return output
        
        return attention_intervention_hook
    
    def apply_attention_intervention(self, target_layer, target_head, newline_positions, intervention_strength=0.8):
        """Apply attention intervention to specific layer and head."""
        
        # Get the target attention layer
        if hasattr(self.model, 'gpt_neox'):  # Pythia/NeoX style
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):  # GPT-style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):  # Other architectures
            layers = self.model.layers
        else:
            print(f"🔍 Model attributes: {dir(self.model)}")
            raise ValueError("Unknown model architecture")
        
        if target_layer >= len(layers):
            raise ValueError(f"Target layer {target_layer} >= number of layers {len(layers)}")
        
        target_layer_module = layers[target_layer]
        
        # Find attention module (model-specific)
        if hasattr(target_layer_module, 'attention'):  # Pythia/NeoX style
            attention_module = target_layer_module.attention
        elif hasattr(target_layer_module, 'attn'):
            attention_module = target_layer_module.attn
        elif hasattr(target_layer_module, 'self_attn'):
            attention_module = target_layer_module.self_attn
        else:
            print(f"🔍 Layer module attributes: {dir(target_layer_module)}")
            raise ValueError("Cannot find attention module in layer")
        
        # Create and register hook
        hook = self.create_attention_hook(target_layer, target_head, newline_positions, intervention_strength)
        handle = attention_module.register_forward_hook(hook)
        
        return handle
    
    def generate_with_intervention(self, input_ids, target_layer, target_head, 
                                 max_new_tokens=100, intervention_strength=0.8):
        """Generate text with attention intervention applied."""
        
        batch_size = input_ids.shape[0]
        
        # Find newline positions in initial input
        newline_positions = self.find_newline_positions(input_ids)
        
        # Apply intervention
        hook_handle = self.apply_attention_intervention(
            target_layer, target_head, newline_positions, intervention_strength
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
    
    def analyze_repetition_induction(self, texts, target_layer, target_head, 
                                   intervention_strength=0.8, max_new_tokens=100):
        """Test whether forcing NEWLINE attention induces repetition."""
        
        results = {
            'baseline_generations': [],
            'intervention_generations': [],
            'baseline_cycles': [],
            'intervention_cycles': [],
            'repetition_induced': [],
            'input_texts': texts
        }
        
        print(f"🧪 Testing attention intervention on layer {target_layer}, head {target_head}")
        print(f"💪 Intervention strength: {intervention_strength}")
        
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
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
            
            # 2. Intervention generation (forced NEWLINE attention)
            intervention_output = self.generate_with_intervention(
                input_ids, target_layer, target_head, max_new_tokens, intervention_strength
            )
            
            # Decode outputs
            baseline_text = self.tokenizer.decode(baseline_output[0], skip_special_tokens=False)
            intervention_text = self.tokenizer.decode(intervention_output[0], skip_special_tokens=False)
            
            # Tokenize outputs for cycle detection
            baseline_tokens = self.tokenizer.encode(baseline_text, return_tensors='pt')[0]
            intervention_tokens = self.tokenizer.encode(intervention_text, return_tensors='pt')[0]
            
            # Detect cycles in both outputs
            baseline_cycles = detect_cycles(baseline_tokens)
            intervention_cycles = detect_cycles(intervention_tokens)
            
            # Check if repetition was induced (detect_cycles returns cycle info or None)
            baseline_cycle_count = 1 if baseline_cycles is not None else 0
            intervention_cycle_count = 1 if intervention_cycles is not None else 0
            repetition_induced = intervention_cycle_count > baseline_cycle_count
            
            # Store results
            results['baseline_generations'].append(baseline_text)
            results['intervention_generations'].append(intervention_text)
            results['baseline_cycles'].append(baseline_cycles)
            results['intervention_cycles'].append(intervention_cycles)
            results['repetition_induced'].append(repetition_induced)
            
            if i < 3:  # Print first few examples
                print(f"\n📝 Example {i+1}:")
                print(f"Input: {text[:100]}...")
                print(f"Baseline cycles: {baseline_cycle_count}")
                print(f"Intervention cycles: {intervention_cycle_count}")
                print(f"Repetition induced: {repetition_induced}")
        
        return results

def create_causal_analysis_plots(results, output_dir, target_layer, target_head):
    """Create visualization plots for the causal analysis."""
    
    # 1. Repetition induction rate
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall repetition induction rate
    ax1 = axes[0, 0]
    induction_rate = np.mean(results['repetition_induced'])
    ax1.bar(['Baseline', 'Intervention'], 
           [0, induction_rate], 
           color=['gray', 'red'], alpha=0.7)
    ax1.set_ylabel('Repetition Induction Rate')
    ax1.set_title(f'Layer {target_layer}, Head {target_head}\nRepetition Induction Rate: {induction_rate:.2%}')
    ax1.set_ylim(0, 1)
    
    # Plot 2: Cycle count comparison
    ax2 = axes[0, 1]
    baseline_counts = [len(cycles) for cycles in results['baseline_cycles']]
    intervention_counts = [len(cycles) for cycles in results['intervention_cycles']]
    
    ax2.scatter(baseline_counts, intervention_counts, alpha=0.6)
    ax2.plot([0, max(max(baseline_counts), max(intervention_counts))], 
             [0, max(max(baseline_counts), max(intervention_counts))], 
             'k--', alpha=0.5, label='No change line')
    ax2.set_xlabel('Baseline Cycle Count')
    ax2.set_ylabel('Intervention Cycle Count')
    ax2.set_title('Cycle Count: Baseline vs Intervention')
    ax2.legend()
    
    # Plot 3: Cycle count distribution
    ax3 = axes[1, 0]
    ax3.hist(baseline_counts, bins=range(max(baseline_counts + intervention_counts) + 2), 
             alpha=0.5, label='Baseline', color='gray')
    ax3.hist(intervention_counts, bins=range(max(baseline_counts + intervention_counts) + 2), 
             alpha=0.5, label='Intervention', color='red')
    ax3.set_xlabel('Number of Cycles')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Cycle Counts')
    ax3.legend()
    
    # Plot 4: Success rate by original cycle count
    ax4 = axes[1, 1]
    baseline_cycle_groups = defaultdict(list)
    for i, baseline_count in enumerate(baseline_counts):
        baseline_cycle_groups[baseline_count].append(results['repetition_induced'][i])
    
    group_keys = sorted(baseline_cycle_groups.keys())
    group_success_rates = [np.mean(baseline_cycle_groups[k]) for k in group_keys]
    
    ax4.plot(group_keys, group_success_rates, 'o-', color='red', markersize=8)
    ax4.set_xlabel('Baseline Cycle Count')
    ax4.set_ylabel('Intervention Success Rate')
    ax4.set_title('Success Rate by Baseline Repetition Level')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Causal Attention Intervention Analysis\nLayer {target_layer}, Head {target_head}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'causal_intervention_L{target_layer}_H{target_head}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_intervention_report(results, output_dir, target_layer, target_head, intervention_strength):
    """Create detailed report of intervention results."""
    
    report_path = output_dir / f'causal_intervention_report_L{target_layer}_H{target_head}.md'
    
    # Calculate summary statistics
    total_texts = len(results['input_texts'])
    induction_rate = np.mean(results['repetition_induced'])
    baseline_cycle_counts = [1 if cycles is not None else 0 for cycles in results['baseline_cycles']]
    intervention_cycle_counts = [1 if cycles is not None else 0 for cycles in results['intervention_cycles']]
    
    avg_baseline_cycles = np.mean(baseline_cycle_counts)
    avg_intervention_cycles = np.mean(intervention_cycle_counts)
    
    # Find successful interventions
    successful_indices = [i for i, induced in enumerate(results['repetition_induced']) if induced]
    
    with open(report_path, 'w') as f:
        f.write(f"# Causal Attention Intervention Report\n\n")
        f.write(f"**Target Layer**: {target_layer}  \n")
        f.write(f"**Target Head**: {target_head}  \n")
        f.write(f"**Intervention Strength**: {intervention_strength}  \n")
        f.write(f"**Total Texts Tested**: {total_texts}  \n\n")
        
        f.write(f"## Summary Results\n\n")
        f.write(f"- **Repetition Induction Rate**: {induction_rate:.2%} ({len(successful_indices)}/{total_texts})\n")
        f.write(f"- **Average Baseline Cycles**: {avg_baseline_cycles:.2f}\n")
        f.write(f"- **Average Intervention Cycles**: {avg_intervention_cycles:.2f}\n")
        f.write(f"- **Cycle Increase**: {avg_intervention_cycles - avg_baseline_cycles:.2f}\n\n")
        
        f.write(f"## Hypothesis Testing\n\n")
        if induction_rate > 0.1:  # 10% threshold
            f.write(f"✅ **HYPOTHESIS SUPPORTED**: Forcing attention to NEWLINE tokens induces repetition at {induction_rate:.1%} rate.\n\n")
        else:
            f.write(f"❌ **HYPOTHESIS NOT SUPPORTED**: Low induction rate ({induction_rate:.1%}) suggests NEWLINE attention is not sufficient for repetition.\n\n")
        
        f.write(f"## Examples of Successful Interventions\n\n")
        for i, idx in enumerate(successful_indices[:3]):  # Show first 3 successful cases
            f.write(f"### Example {i+1}\n")
            f.write(f"**Input**: {results['input_texts'][idx][:100]}...\n\n")
            f.write(f"**Baseline** ({len(results['baseline_cycles'][idx])} cycles): \n")
            f.write(f"```\n{results['baseline_generations'][idx][:200]}...\n```\n\n")
            f.write(f"**Intervention** ({len(results['intervention_cycles'][idx])} cycles): \n")
            f.write(f"```\n{results['intervention_generations'][idx][:200]}...\n```\n\n")
        
        f.write(f"## Scientific Implications\n\n")
        f.write(f"1. **Causal Role**: {'Confirmed' if induction_rate > 0.1 else 'Not confirmed'} causal role of NEWLINE attention in repetition\n")
        f.write(f"2. **Mechanism**: {'Direct' if induction_rate > 0.3 else 'Indirect' if induction_rate > 0.1 else 'Minimal'} effect of attention manipulation\n")
        f.write(f"3. **Sufficiency**: NEWLINE attention is {'sufficient' if induction_rate > 0.5 else 'partially sufficient' if induction_rate > 0.2 else 'not sufficient'} for repetition induction\n")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Causal Attention Intervention Experiment")
    parser.add_argument("--layer", type=int, default=19, help="Target layer for intervention")
    parser.add_argument("--head", type=int, default=0, help="Target head for intervention")
    parser.add_argument("--strength", type=float, default=0.8, help="Intervention strength")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Causal Attention Intervention Experiment")
    print(f"📋 Parameters:")
    print(f"   - Target layer: {args.layer}")
    print(f"   - Target head: {args.head}")
    print(f"   - Intervention strength: {args.strength}")
    print(f"   - Number of samples: {args.n_samples}")
    print(f"   - Checkpoint: {args.checkpoint}")
    
    # Load model and tokenizer
    print(f"🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=args.checkpoint)
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Load non-repeating text dataset
    print(f"📚 Loading dataset...")
    dataset = load_text_dataset("JeanKaddour/minipile")  # Load non-repeating texts
    test_texts = dataset[:args.n_samples]
    
    print(f"📊 Loaded {len(test_texts)} non-repeating texts for intervention testing")
    
    # Create output directory
    output_dir = Path(f"./plots/causal_intervention_L{args.layer}_H{args.head}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize intervention processor
    processor = AttentionInterventionProcessor(model, tokenizer)
    
    # Run causal intervention experiment
    print(f"🧪 Running causal intervention experiment...")
    results = processor.analyze_repetition_induction(
        test_texts, args.layer, args.head, args.strength
    )
    
    # Create analysis plots
    print(f"📊 Creating analysis plots...")
    plot_path = create_causal_analysis_plots(results, output_dir, args.layer, args.head)
    print(f"   ✅ Plots saved: {plot_path}")
    
    # Create detailed report
    print(f"📝 Creating analysis report...")
    report_path = create_intervention_report(results, output_dir, args.layer, args.head, args.strength)
    print(f"   ✅ Report saved: {report_path}")
    
    # Save raw results
    results_path = output_dir / f"causal_intervention_results_L{args.layer}_H{args.head}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Print summary
    induction_rate = np.mean(results['repetition_induced'])
    print(f"\n🎯 Experiment Summary:")
    print(f"   - Repetition induction rate: {induction_rate:.2%}")
    print(f"   - Successful interventions: {sum(results['repetition_induced'])}/{len(results['repetition_induced'])}")
    
    if induction_rate > 0.1:
        print(f"   ✅ HYPOTHESIS SUPPORTED: NEWLINE attention causally induces repetition")
    else:
        print(f"   ❌ HYPOTHESIS NOT SUPPORTED: NEWLINE attention does not reliably induce repetition")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()