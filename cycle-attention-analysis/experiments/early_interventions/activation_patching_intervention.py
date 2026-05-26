#!/usr/bin/env python3
"""
Activation Patching Causal Intervention Experiment
Tests whether patching activations from repetitive texts can induce repetition.
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
print("✅ Basic imports done")

# Import from the original intervention script
from causal_attention_intervention import AttentionInterventionProcessor
from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from modules.cached_data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles

print("✅ All imports successful!")

class ActivationPatchingProcessor(AttentionInterventionProcessor):
    """Implements activation patching for causal intervention experiments."""
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.cached_activations = {}
        
    def cache_activations(self, input_ids, layer_idx, component='attention'):
        """Cache activations from a forward pass for later patching."""
        cached_acts = {}
        
        def cache_hook(module, input, output):
            if component == 'attention':
                # For attention, cache the attention weights
                if isinstance(output, tuple) and len(output) >= 2:
                    hidden_states, attention_weights = output
                    if attention_weights is not None:
                        cached_acts['attention_weights'] = attention_weights.detach().clone()
                        cached_acts['hidden_states'] = hidden_states.detach().clone()
            elif component == 'mlp':
                # For MLP, cache the output activations
                cached_acts['mlp_output'] = output.detach().clone()
            else:
                # For general activations
                cached_acts['activations'] = output.detach().clone()
        
        # Get the target layer
        if hasattr(self.model, 'gpt_neox'):  # Pythia/NeoX style
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        target_layer = layers[layer_idx]
        
        # Choose component to cache
        if component == 'attention':
            if hasattr(target_layer, 'attention'):
                hook_module = target_layer.attention
            elif hasattr(target_layer, 'attn'):
                hook_module = target_layer.attn
            elif hasattr(target_layer, 'self_attn'):
                hook_module = target_layer.self_attn
            else:
                raise ValueError("Cannot find attention module")
        elif component == 'mlp':
            if hasattr(target_layer, 'mlp'):
                hook_module = target_layer.mlp
            elif hasattr(target_layer, 'feed_forward'):
                hook_module = target_layer.feed_forward
            else:
                raise ValueError("Cannot find MLP module")
        else:
            hook_module = target_layer
        
        # Register hook and run forward pass
        handle = hook_module.register_forward_hook(cache_hook)
        
        try:
            with torch.no_grad():
                _ = self.model(input_ids)
        finally:
            handle.remove()
        
        return cached_acts
    
    def create_activation_patch_hook(self, cached_activations, component='attention'):
        """Create a hook that patches in cached activations."""
        
        def patch_hook(module, input, output):
            if component == 'attention' and 'attention_weights' in cached_activations:
                # Patch attention weights
                if isinstance(output, tuple) and len(output) >= 2:
                    hidden_states, attention_weights = output
                    if attention_weights is not None:
                        # Use cached attention weights
                        patched_attention = cached_activations['attention_weights']
                        # Ensure batch dimensions match
                        if patched_attention.shape[0] == attention_weights.shape[0]:
                            return (hidden_states, patched_attention)
                return output
            elif component == 'mlp' and 'mlp_output' in cached_activations:
                # Patch MLP output
                patched_output = cached_activations['mlp_output']
                if patched_output.shape[0] == output.shape[0]:
                    return patched_output
                return output
            else:
                # Patch general activations
                if 'activations' in cached_activations:
                    patched_activations = cached_activations['activations']
                    if patched_activations.shape[0] == output.shape[0]:
                        return patched_activations
                return output
        
        return patch_hook
    
    def generate_with_activation_patching(self, input_ids, layer_idx, cached_activations, 
                                        component='attention', max_new_tokens=100):
        """Generate text with activation patching applied."""
        
        # Get the target layer and component
        if hasattr(self.model, 'gpt_neox'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        target_layer = layers[layer_idx]
        
        # Choose component to patch
        if component == 'attention':
            if hasattr(target_layer, 'attention'):
                hook_module = target_layer.attention
            elif hasattr(target_layer, 'attn'):
                hook_module = target_layer.attn
            elif hasattr(target_layer, 'self_attn'):
                hook_module = target_layer.self_attn
            else:
                raise ValueError("Cannot find attention module")
        elif component == 'mlp':
            if hasattr(target_layer, 'mlp'):
                hook_module = target_layer.mlp
            elif hasattr(target_layer, 'feed_forward'):
                hook_module = target_layer.feed_forward
            else:
                raise ValueError("Cannot find MLP module")
        else:
            hook_module = target_layer
        
        # Create and register patch hook
        patch_hook = self.create_activation_patch_hook(cached_activations, component)
        handle = hook_module.register_forward_hook(patch_hook)
        
        try:
            # Generate with patched activations
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
            handle.remove()
        
        return generated
    
    def analyze_activation_patching_induction(self, non_rep_texts, rep_texts, layer_idx, 
                                            component='attention', max_new_tokens=100):
        """Test whether patching activations from repetitive texts induces repetition."""
        
        results = {
            'baseline_generations': [],
            'patched_generations': [],
            'baseline_cycles': [],
            'patched_cycles': [],
            'repetition_induced': [],
            'non_rep_inputs': non_rep_texts,
            'rep_sources': rep_texts,
            'component': component
        }
        
        print(f"🧪 Testing activation patching on layer {layer_idx}")
        print(f"🔧 Component: {component}")
        print(f"📊 Non-repetitive inputs: {len(non_rep_texts)}")
        print(f"🔄 Repetitive sources: {len(rep_texts)}")
        
        # First, cache activations from repetitive texts
        print("📦 Caching activations from repetitive texts...")
        rep_activations = []
        for rep_text in tqdm(rep_texts[:len(non_rep_texts)], desc="Caching"):
            inputs = self.tokenizer(rep_text, return_tensors='pt', padding=True, 
                                  truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.device)
            
            cached_acts = self.cache_activations(input_ids, layer_idx, component)
            rep_activations.append(cached_acts)
        
        # Now test patching on non-repetitive texts
        print("🧪 Testing activation patching...")
        for i, (non_rep_text, cached_acts) in enumerate(tqdm(
            zip(non_rep_texts, rep_activations), desc="Processing", 
            total=min(len(non_rep_texts), len(rep_activations))
        )):
            # Tokenize non-repetitive input
            inputs = self.tokenizer(non_rep_text, return_tensors='pt', padding=True, 
                                  truncation=True, max_length=512)
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
            
            # 2. Activation patching generation
            patched_output = self.generate_with_activation_patching(
                input_ids, layer_idx, cached_acts, component, max_new_tokens
            )
            
            # Decode outputs
            baseline_text = self.tokenizer.decode(baseline_output[0], skip_special_tokens=False)
            patched_text = self.tokenizer.decode(patched_output[0], skip_special_tokens=False)
            
            # Tokenize outputs for cycle detection
            baseline_tokens = self.tokenizer.encode(baseline_text, return_tensors='pt')[0]
            patched_tokens = self.tokenizer.encode(patched_text, return_tensors='pt')[0]
            
            # Detect cycles
            baseline_cycles = detect_cycles(baseline_tokens)
            patched_cycles = detect_cycles(patched_tokens)
            
            # Check if repetition was induced
            baseline_cycle_count = 1 if baseline_cycles is not None else 0
            patched_cycle_count = 1 if patched_cycles is not None else 0
            repetition_induced = patched_cycle_count > baseline_cycle_count
            
            # Store results
            results['baseline_generations'].append(baseline_text)
            results['patched_generations'].append(patched_text)
            results['baseline_cycles'].append(baseline_cycles)
            results['patched_cycles'].append(patched_cycles)
            results['repetition_induced'].append(repetition_induced)
            
            if i < 3:  # Print first few examples
                print(f"\\n📝 Example {i+1}:")
                print(f"Non-rep input: {non_rep_text[:100]}...")
                print(f"Baseline cycles: {baseline_cycle_count}")
                print(f"Patched cycles: {patched_cycle_count}")
                print(f"Repetition induced: {repetition_induced}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Activation Patching Causal Intervention Experiment")
    parser.add_argument("--layer", type=int, default=19, help="Target layer for patching")
    parser.add_argument("--component", type=str, default="attention", 
                       choices=["attention", "mlp", "layer"], help="Component to patch")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Activation Patching Causal Intervention Experiment")
    print(f"📋 Parameters:")
    print(f"   - Target layer: {args.layer}")
    print(f"   - Component: {args.component}")
    print(f"   - Number of samples: {args.n_samples}")
    
    # Load model and tokenizer
    print(f"🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=args.checkpoint)
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Load datasets
    print(f"📚 Loading datasets...")
    
    # Load non-repetitive texts
    try:
        non_rep_dataset = load_text_dataset("JeanKaddour/minipile")
        non_repetitive_texts = non_rep_dataset[:10]
    except Exception as e:
        print("Failed to load JeanKaddour/minipile dataset, using fallback...")
        non_rep_dataset = load_text_dataset("JeanKaddour/minipile")
    
    # Load repetitive texts  
    try:
        rep_dataset = load_text_dataset("cycle")
        rep_texts = rep_dataset[:args.n_samples]
    except:
        print("Failed to load cycle dataset, generating synthetic repetitive texts...")
        rep_texts = [
            "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
            "Hello world! Hello world! Hello world! Hello world!",
            "Testing repetition. Testing repetition. Testing repetition.",
            "A B C D. A B C D. A B C D. A B C D.",
            "Repeat this phrase. Repeat this phrase. Repeat this phrase."
        ][:args.n_samples]
    
    print(f"📊 Loaded {len(non_rep_texts)} non-repetitive and {len(rep_texts)} repetitive texts")
    
    # Create output directory
    output_dir = Path(f"./plots/activation_patching_L{args.layer}_{args.component}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize activation patching processor
    processor = ActivationPatchingProcessor(model, tokenizer)
    
    # Run activation patching experiment
    print(f"🧪 Running activation patching experiment...")
    results = processor.analyze_activation_patching_induction(
        non_rep_texts, rep_texts, args.layer, args.component
    )
    
    # Save raw results
    results_path = output_dir / f"activation_patching_results_L{args.layer}_{args.component}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Create report
    induction_rate = np.mean(results['repetition_induced'])
    report_path = output_dir / f"activation_patching_report_L{args.layer}_{args.component}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Activation Patching Causal Intervention Report\\n\\n")
        f.write(f"**Target Layer**: {args.layer}  \\n")
        f.write(f"**Component**: {args.component}  \\n")
        f.write(f"**Total Tests**: {len(results['repetition_induced'])}  \\n\\n")
        
        f.write(f"## Summary Results\\n\\n")
        f.write(f"- **Repetition Induction Rate**: {induction_rate:.2%} ({sum(results['repetition_induced'])}/{len(results['repetition_induced'])})\\n")
        
        if induction_rate > 0.2:
            f.write(f"\\n✅ **HYPOTHESIS SUPPORTED**: Activation patching induces repetition at {induction_rate:.1%} rate.\\n\\n")
        else:
            f.write(f"\\n❌ **HYPOTHESIS NOT SUPPORTED**: Low induction rate ({induction_rate:.1%}) suggests patched activations are insufficient.\\n\\n")
        
        f.write(f"## Mechanism Insights\\n\\n")
        f.write(f"- **Patching approach**: Transfer {args.component} activations from repetitive to non-repetitive contexts\\n")
        f.write(f"- **Causal effect**: {'Strong' if induction_rate > 0.2 else 'Weak to None'}\\n")
        f.write(f"- **Implication**: {'Repetitive patterns are encoded in {}'.format(args.component) if induction_rate > 0.2 else 'Repetitive patterns require complex multi-component coordination'}\\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\\n🎯 Activation Patching Summary:")
    print(f"   - Repetition induction rate: {induction_rate:.2%}")
    print(f"   - Successful patches: {sum(results['repetition_induced'])}/{len(results['repetition_induced'])}")
    print(f"   - Component tested: {args.component}")
    
    if induction_rate > 0.2:
        print(f"   ✅ CAUSAL EVIDENCE: {args.component} activations drive repetition")
    else:
        print(f"   ❌ NO CAUSAL EVIDENCE: {args.component} patching ineffective")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()