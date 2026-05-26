#!/usr/bin/env python3
"""
Multi-Head Causal Attention Intervention Experiment
Tests whether coordinated NEWLINE attention across multiple heads can induce repetition.
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

class MultiHeadAttentionInterventionProcessor(AttentionInterventionProcessor):
    """Extends the base processor to handle multiple heads simultaneously."""
    
    def create_multi_head_attention_hook(self, target_layer, target_heads, newline_positions, intervention_strength=0.8):
        """Create a hook that forces attention to NEWLINE tokens across multiple heads."""
        
        def multi_head_attention_intervention_hook(module, input, output):
            # GPT-NeoX returns (hidden_states, attention_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, attention_weights = output
            else:
                return output
                
            if attention_weights is None:
                return output
                
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Modify all target heads
            for target_head in target_heads:
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
            return (hidden_states, attention_weights)
        
        return multi_head_attention_intervention_hook
    
    def apply_multi_head_intervention(self, target_layer, target_heads, newline_positions, intervention_strength=0.8):
        """Apply attention intervention to specific layer and multiple heads."""
        
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
        hook = self.create_multi_head_attention_hook(target_layer, target_heads, newline_positions, intervention_strength)
        handle = attention_module.register_forward_hook(hook)
        
        return handle
    
    def generate_with_multi_head_intervention(self, input_ids, target_layer, target_heads, 
                                            max_new_tokens=100, intervention_strength=0.8):
        """Generate text with multi-head attention intervention applied."""
        
        # Find newline positions in initial input
        newline_positions = self.find_newline_positions(input_ids)
        
        # Apply multi-head intervention
        hook_handle = self.apply_multi_head_intervention(
            target_layer, target_heads, newline_positions, intervention_strength
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
    
    def analyze_multi_head_repetition_induction(self, texts, target_layer, target_heads, 
                                              intervention_strength=0.8, max_new_tokens=100):
        """Test whether forcing NEWLINE attention across multiple heads induces repetition."""
        
        results = {
            'baseline_generations': [],
            'intervention_generations': [],
            'baseline_cycles': [],
            'intervention_cycles': [],
            'repetition_induced': [],
            'input_texts': texts,
            'target_heads': target_heads
        }
        
        print(f"🧪 Testing multi-head attention intervention on layer {target_layer}")
        print(f"🎯 Target heads: {target_heads}")
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
            
            # 2. Multi-head intervention generation
            intervention_output = self.generate_with_multi_head_intervention(
                input_ids, target_layer, target_heads, max_new_tokens, intervention_strength
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
            
            # Check if repetition was induced
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
                print(f"\\n📝 Example {i+1}:")
                print(f"Input: {text[:100]}...")
                print(f"Baseline cycles: {baseline_cycle_count}")
                print(f"Intervention cycles: {intervention_cycle_count}")
                print(f"Repetition induced: {repetition_induced}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Multi-Head Causal Attention Intervention Experiment")
    parser.add_argument("--layer", type=int, default=19, help="Target layer for intervention")
    parser.add_argument("--heads", type=str, default="0,1,2,3", help="Target heads (comma-separated)")
    parser.add_argument("--strength", type=float, default=0.8, help="Intervention strength")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    # Parse target heads
    target_heads = [int(h.strip()) for h in args.heads.split(',')]
    
    print(f"🚀 Starting Multi-Head Causal Attention Intervention Experiment")
    print(f"📋 Parameters:")
    print(f"   - Target layer: {args.layer}")
    print(f"   - Target heads: {target_heads}")
    print(f"   - Intervention strength: {args.strength}")
    print(f"   - Number of samples: {args.n_samples}")
    
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
    heads_str = "_".join(map(str, target_heads))
    output_dir = Path(f"./plots/multi_head_intervention_L{args.layer}_H{heads_str}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize multi-head intervention processor
    processor = MultiHeadAttentionInterventionProcessor(model, tokenizer)
    
    # Run multi-head causal intervention experiment
    print(f"🧪 Running multi-head causal intervention experiment...")
    results = processor.analyze_multi_head_repetition_induction(
        test_texts, args.layer, target_heads, args.strength
    )
    
    # Save raw results
    results_path = output_dir / f"multi_head_intervention_results_L{args.layer}_H{heads_str}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Create report
    induction_rate = np.mean(results['repetition_induced'])
    report_path = output_dir / f"multi_head_intervention_report_L{args.layer}_H{heads_str}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Multi-Head Causal Attention Intervention Report\\n\\n")
        f.write(f"**Target Layer**: {args.layer}  \\n")
        f.write(f"**Target Heads**: {target_heads}  \\n")
        f.write(f"**Intervention Strength**: {args.strength}  \\n")
        f.write(f"**Total Texts Tested**: {len(test_texts)}  \\n\\n")
        
        f.write(f"## Summary Results\\n\\n")
        f.write(f"- **Repetition Induction Rate**: {induction_rate:.2%} ({sum(results['repetition_induced'])}/{len(results['repetition_induced'])})\\n")
        
        if induction_rate > 0.1:
            f.write(f"\\n✅ **HYPOTHESIS SUPPORTED**: Multi-head NEWLINE attention coordination induces repetition at {induction_rate:.1%} rate.\\n\\n")
        else:
            f.write(f"\\n❌ **HYPOTHESIS NOT SUPPORTED**: Low induction rate ({induction_rate:.1%}) suggests coordinated NEWLINE attention is not sufficient.\\n\\n")
        
        f.write(f"## Comparison with Single-Head Results\\n\\n")
        f.write(f"- **Multi-head approach**: {len(target_heads)} heads simultaneously\\n")
        f.write(f"- **Coordination effect**: {'Positive' if induction_rate > 0.1 else 'Minimal'}\\n")
        f.write(f"- **Mechanism insight**: {'Coordination improves efficacy' if induction_rate > 0.1 else 'Coordination insufficient for induction'}\\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\\n🎯 Multi-Head Experiment Summary:")
    print(f"   - Repetition induction rate: {induction_rate:.2%}")
    print(f"   - Successful interventions: {sum(results['repetition_induced'])}/{len(results['repetition_induced'])}")
    print(f"   - Coordinated heads: {len(target_heads)}")
    
    if induction_rate > 0.1:
        print(f"   ✅ COORDINATION EFFECT: Multi-head intervention shows promise")
    else:
        print(f"   ❌ NO COORDINATION EFFECT: Multi-head intervention ineffective")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()