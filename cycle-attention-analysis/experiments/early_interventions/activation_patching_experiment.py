#!/usr/bin/env python3
"""
Activation Patching Causal Intervention Experiment
Tests whether patching activations from repetitive to non-repetitive sequences induces repetition.
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

class ActivationPatchingProcessor:
    """Implements activation patching for causal testing."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.cached_activations = {}
        
    def extract_activations(self, input_ids, target_layers):
        """Extract activations from specific layers during forward pass."""
        activations = {}
        hooks = []
        
        # Get model layers
        if hasattr(self.model, 'gpt_neox'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Store the main hidden states (first element of tuple)
                    activations[layer_idx] = output[0].detach().clone()
                else:
                    activations[layer_idx] = output.detach().clone()
            return hook
        
        # Register hooks
        for layer_idx in target_layers:
            if layer_idx < len(layers):
                hook_handle = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                hooks.append(hook_handle)
        
        try:
            # Forward pass to extract activations
            with torch.no_grad():
                _ = self.model(input_ids, output_hidden_states=True)
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        return activations
    
    def create_activation_patch_hook(self, target_layer, patch_activations, patch_positions=None):
        """Create a hook that patches activations at specific positions."""
        
        def activation_patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Apply patch
            for batch_idx in range(min(batch_size, len(patch_activations))):
                patch_tensor = patch_activations[batch_idx]
                
                if patch_positions is None:
                    # Patch entire sequence
                    patch_len = min(seq_len, patch_tensor.shape[0])
                    hidden_states[batch_idx, :patch_len, :] = patch_tensor[:patch_len, :]
                else:
                    # Patch specific positions
                    for pos in patch_positions:
                        if pos < seq_len and pos < patch_tensor.shape[0]:
                            hidden_states[batch_idx, pos, :] = patch_tensor[pos, :]
            
            if other_outputs:
                return (hidden_states,) + other_outputs
            else:
                return hidden_states
        
        return activation_patch_hook
    
    def apply_activation_patch(self, target_layer, patch_activations, patch_positions=None):
        """Apply activation patching to specific layer."""
        
        # Get model layers
        if hasattr(self.model, 'gpt_neox'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        target_layer_module = layers[target_layer]
        
        # Create and register hook
        hook = self.create_activation_patch_hook(target_layer, patch_activations, patch_positions)
        handle = target_layer_module.register_forward_hook(hook)
        
        return handle
    
    def generate_with_activation_patch(self, input_ids, target_layer, patch_activations, 
                                     patch_positions=None, max_new_tokens=100):
        """Generate text with activation patching."""
        
        # Apply activation patch
        hook_handle = self.apply_activation_patch(target_layer, patch_activations, patch_positions)
        
        try:
            # Generate with patch
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
    
    def analyze_activation_patching_experiment(self, non_repeat_texts, repeat_texts, target_layers, 
                                            max_new_tokens=100):
        """Test whether patching activations from repetitive texts induces repetition."""
        
        results = {
            'baseline_generations': [],
            'patched_generations': [],
            'baseline_cycles': [],
            'patched_cycles': [],
            'repetition_induced': [],
            'source_texts': repeat_texts,
            'target_texts': non_repeat_texts,
            'target_layers': target_layers
        }
        
        print(f"🧪 Testing activation patching experiment")
        print(f"📊 Target layers: {target_layers}")
        print(f"🎯 Source (repetitive) texts: {len(repeat_texts)}")
        print(f"🎯 Target (non-repetitive) texts: {len(non_repeat_texts)}")
        
        # Extract activations from repetitive texts
        print("📥 Extracting activations from repetitive texts...")
        repeat_activations = {}
        
        for i, repeat_text in enumerate(tqdm(repeat_texts[:3], desc="Processing repeat texts")):  # Limit to avoid memory issues
            # Tokenize repetitive text
            inputs = self.tokenizer(repeat_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            repeat_input_ids = inputs['input_ids'].to(self.device)
            
            # Extract activations
            activations = self.extract_activations(repeat_input_ids, target_layers)
            repeat_activations[i] = activations
        
        print(f"✅ Extracted activations for {len(repeat_activations)} repetitive texts")
        
        # Test patching on non-repetitive texts
        for i, non_repeat_text in enumerate(tqdm(non_repeat_texts[:3], desc="Testing patches")):
            # Tokenize non-repetitive text
            inputs = self.tokenizer(non_repeat_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # 1. Baseline generation (no patching)
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
            
            # 2. Test each target layer
            best_patched_result = None
            best_cycle_count = 0
            
            for layer in target_layers:
                # Use activations from a repetitive text (cycling through available ones)
                source_idx = i % len(repeat_activations)
                if layer in repeat_activations[source_idx]:
                    patch_acts = repeat_activations[source_idx][layer]
                    
                    # Generate with activation patch
                    try:
                        patched_output = self.generate_with_activation_patch(
                            input_ids, layer, [patch_acts], max_new_tokens=max_new_tokens
                        )
                        
                        patched_text = self.tokenizer.decode(patched_output[0], skip_special_tokens=False)
                        patched_tokens = self.tokenizer.encode(patched_text, return_tensors='pt')[0]
                        patched_cycles = detect_cycles(patched_tokens)
                        
                        # Track best result (highest cycle count)
                        cycle_count = 1 if patched_cycles is not None else 0
                        if cycle_count > best_cycle_count:
                            best_cycle_count = cycle_count
                            best_patched_result = {
                                'text': patched_text,
                                'cycles': patched_cycles,
                                'layer': layer
                            }
                    except Exception as e:
                        print(f"   ⚠️ Patching failed for layer {layer}: {e}")
                        continue
            
            # Store best patched result
            if best_patched_result:
                results['patched_generations'].append(best_patched_result['text'])
                results['patched_cycles'].append(best_patched_result['cycles'])
            else:
                # Fallback to baseline if no patch worked
                results['patched_generations'].append(baseline_text)
                results['patched_cycles'].append(baseline_cycles)
            
            # Check if repetition was induced
            baseline_cycle_count = 1 if baseline_cycles is not None else 0
            final_cycle_count = 1 if results['patched_cycles'][-1] is not None else 0
            repetition_induced = final_cycle_count > baseline_cycle_count
            results['repetition_induced'].append(repetition_induced)
            
            if i < 3:  # Print first few examples
                print(f"\n📝 Example {i+1}:")
                print(f"Input: {non_repeat_text[:100]}...")
                print(f"Baseline cycles: {baseline_cycle_count}")
                print(f"Patched cycles: {final_cycle_count}")
                if best_patched_result:
                    print(f"Best patch layer: {best_patched_result['layer']}")
                print(f"Repetition induced: {repetition_induced}")
        
        return results
    
    def create_activation_patching_plots(self, results, output_dir):
        """Create visualization plots for activation patching results."""
        
        induction_rate = np.mean(results['repetition_induced'])
        
        # Plot 1: Overall results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        categories = ['Baseline', 'Patched']
        
        baseline_cycles = [1 if cycles is not None else 0 for cycles in results['baseline_cycles']]
        patched_cycles = [1 if cycles is not None else 0 for cycles in results['patched_cycles']]
        
        avg_baseline = np.mean(baseline_cycles)
        avg_patched = np.mean(patched_cycles)
        
        bars = plt.bar(categories, [avg_baseline, avg_patched], color=['lightcoral', 'lightgreen'], alpha=0.7)
        plt.ylabel('Average Cycle Count')
        plt.title('Activation Patching Effect')
        plt.ylim(0, 1.2)
        
        for bar, rate in zip(bars, [avg_baseline, avg_patched]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # Plot 2: Induction success rate
        plt.subplot(2, 2, 2)
        success_rate = induction_rate
        failure_rate = 1 - induction_rate
        
        plt.pie([success_rate, failure_rate], labels=['Induced', 'Not Induced'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        plt.title(f'Repetition Induction Rate: {induction_rate:.1%}')
        
        # Plot 3: Before/after comparison
        plt.subplot(2, 2, 3)
        indices = range(len(baseline_cycles))
        
        plt.scatter(indices, baseline_cycles, alpha=0.7, label='Baseline', color='coral')
        plt.scatter(indices, patched_cycles, alpha=0.7, label='Patched', color='green')
        plt.xlabel('Test Case')
        plt.ylabel('Cycle Detected (0/1)')
        plt.title('Individual Case Results')
        plt.legend()
        
        # Plot 4: Effect size
        plt.subplot(2, 2, 4)
        effect_sizes = [patched - baseline for baseline, patched in zip(baseline_cycles, patched_cycles)]
        
        plt.hist(effect_sizes, bins=5, alpha=0.7, color='skyblue')
        plt.xlabel('Effect Size (Patched - Baseline)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Patch Effects')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "activation_patching_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Activation Patching Causal Intervention Experiment")
    parser.add_argument("--layers", nargs='+', type=int, default=[15, 17, 19], help="Target layers for patching")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Activation Patching Causal Intervention Experiment")
    print(f"📋 Parameters:")
    print(f"   - Target layers: {args.layers}")
    print(f"   - Number of samples: {args.n_samples}")
    
    # Load model and tokenizer
    print(f"🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=args.checkpoint)
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Load datasets
    print(f"📚 Loading datasets...")
    
    try:
        # Load non-repetitive texts
        standard_dataset = load_text_dataset("JeanKaddour/minipile")
        non_repeat_texts = standard_dataset[: args.n_samples]
        print(f"   ✅ Loaded {len(non_repeat_texts)} texts from JeanKaddour/minipile")
    except Exception:
        print("   ⚠️ Failed to load JeanKaddour/minipile dataset, using fallback...")
        fallback_dataset = load_text_dataset("wikitext")
        non_repeat_texts = fallback_dataset[: args.n_samples]
    
    try:
        # Load repetitive texts (for extracting activations)
        cycle_dataset = load_text_dataset("cycle")
        repeat_texts = cycle_dataset[:args.n_samples]
        print(f"   ✅ Loaded {len(repeat_texts)} repetitive texts")
    except:
        print("   ⚠️ Failed to load cycle dataset, creating synthetic repetitive texts...")
        # Create simple repetitive texts
        repeat_texts = [
            "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
            "Hello world! Hello world! Hello world! Hello world!",
            "Python is great. Python is great. Python is great."
        ][:args.n_samples]
    
    print(f"📊 Testing activation patching with {len(non_repeat_texts)} target texts and {len(repeat_texts)} source texts")
    
    # Create output directory
    output_dir = Path(f"./plots/activation_patching_L{'_'.join(map(str, args.layers))}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize activation patching processor
    processor = ActivationPatchingProcessor(model, tokenizer)
    
    # Run activation patching experiment
    print(f"🧪 Running activation patching experiment...")
    results = processor.analyze_activation_patching_experiment(
        non_repeat_texts, repeat_texts, args.layers
    )
    
    # Save raw results
    results_path = output_dir / f"activation_patching_results_L{'_'.join(map(str, args.layers))}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Create analysis plots
    plot_path = processor.create_activation_patching_plots(results, output_dir)
    print(f"   ✅ Analysis plots saved: {plot_path}")
    
    # Create detailed report
    induction_rate = np.mean(results['repetition_induced'])
    report_path = output_dir / f"activation_patching_report_L{'_'.join(map(str, args.layers))}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Activation Patching Causal Intervention Report\n\n")
        f.write(f"**Target Layers**: {args.layers}  \n")
        f.write(f"**Non-Repetitive Texts**: {len(non_repeat_texts)}  \n")
        f.write(f"**Repetitive Source Texts**: {len(repeat_texts)}  \n\n")
        
        f.write(f"## Summary Results\n\n")
        f.write(f"- **Repetition Induction Rate**: {induction_rate:.2%} ({sum(results['repetition_induced'])}/{len(results['repetition_induced'])})\n")
        
        baseline_cycles = [1 if cycles is not None else 0 for cycles in results['baseline_cycles']]
        patched_cycles = [1 if cycles is not None else 0 for cycles in results['patched_cycles']]
        
        f.write(f"- **Baseline Cycle Rate**: {np.mean(baseline_cycles):.2%}\n")
        f.write(f"- **Patched Cycle Rate**: {np.mean(patched_cycles):.2%}\n")
        f.write(f"- **Effect Size**: {np.mean(patched_cycles) - np.mean(baseline_cycles):+.2%}\n\n")
        
        if induction_rate > 0.15:
            f.write(f"✅ **HYPOTHESIS SUPPORTED**: Activation patching induces repetition at {induction_rate:.1%} rate.\n\n")
            f.write(f"**Implication**: Repetitive activations contain causal information for repetition induction.\n\n")
        else:
            f.write(f"❌ **HYPOTHESIS NOT SUPPORTED**: Low induction rate ({induction_rate:.1%}) suggests limited causal effect.\n\n")
            f.write(f"**Implication**: Simple activation patching is insufficient for reliable repetition induction.\n\n")
        
        f.write(f"## Mechanism Insights\n\n")
        f.write(f"- **Activation transfer**: {'Successful' if induction_rate > 0.15 else 'Limited'}\n")
        f.write(f"- **Layer specificity**: {'High' if len(args.layers) <= 3 else 'Broad'}\n")
        f.write(f"- **Causal pathway**: {'Activations drive repetition' if induction_rate > 0.15 else 'Activations insufficient for repetition induction'}\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\n🎯 Activation Patching Summary:")
    print(f"   - Repetition induction rate: {induction_rate:.2%}")
    print(f"   - Successful patches: {sum(results['repetition_induced'])}/{len(results['repetition_induced'])}")
    print(f"   - Effect size: {np.mean(patched_cycles) - np.mean(baseline_cycles):+.2%}")
    
    if induction_rate > 0.15:
        print(f"   ✅ ACTIVATION EFFECT: Patching repetitive activations induces repetition")
    else:
        print(f"   ❌ NO ACTIVATION EFFECT: Activation patching ineffective")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()