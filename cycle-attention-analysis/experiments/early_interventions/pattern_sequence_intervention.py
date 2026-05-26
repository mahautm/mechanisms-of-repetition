#!/usr/bin/env python3
"""
Emerging Pattern Sequence Causal Intervention Experiment
Tests whether forcing attention to detected emerging patterns can induce repetition.
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
import re
print("✅ Basic imports done")

# Import from the original intervention script
from causal_attention_intervention import AttentionInterventionProcessor
from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from modules.cached_data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles

print("✅ All imports successful!")

class EmergingPatternProcessor(AttentionInterventionProcessor):
    """Implements pattern detection and intervention for emerging sequences."""
    
    def detect_emerging_patterns(self, tokens, window_size=3, min_frequency=2):
        """Detect emerging patterns in token sequences."""
        patterns = defaultdict(list)  # pattern -> [positions]
        
        # Convert tokens to list if tensor
        if torch.is_tensor(tokens):
            tokens = tokens.cpu().tolist()
        
        # Extract n-grams of various sizes
        for n in range(2, window_size + 1):
            for i in range(len(tokens) - n + 1):
                pattern = tuple(tokens[i:i+n])
                patterns[pattern].append(i)
        
        # Filter patterns by frequency and detect emerging ones
        emerging_patterns = {}
        for pattern, positions in patterns.items():
            if len(positions) >= min_frequency:
                # Check if pattern appears to be emerging (positions getting closer)
                if len(positions) >= 2:
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    avg_gap = sum(gaps) / len(gaps)
                    # Pattern is "emerging" if gaps are decreasing or small
                    if avg_gap <= 10 or (len(gaps) > 1 and gaps[-1] < gaps[0]):
                        emerging_patterns[pattern] = {
                            'positions': positions,
                            'frequency': len(positions),
                            'avg_gap': avg_gap,
                            'last_position': positions[-1]
                        }
        
        return emerging_patterns
    
    def create_pattern_attention_hook(self, target_layer, pattern_positions, intervention_strength=0.8):
        """Create a hook that forces attention to detected pattern positions."""
        
        def pattern_attention_intervention_hook(module, input, output):
            # GPT-NeoX returns (hidden_states, attention_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, attention_weights = output
            else:
                return output
                
            if attention_weights is None:
                return output
                
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Apply intervention to all heads for now
            for batch_idx in range(batch_size):
                if batch_idx < len(pattern_positions) and pattern_positions[batch_idx]:
                    positions = pattern_positions[batch_idx]
                    
                    for head_idx in range(num_heads):
                        for pos in positions:
                            if pos < seq_len:
                                # Increase attention from current and nearby positions to pattern position
                                for query_pos in range(max(0, pos-5), min(seq_len, pos+5)):
                                    attention_weights[batch_idx, head_idx, query_pos, pos] *= (1 + intervention_strength)
                        
                        # Renormalize attention weights for this head
                        attention_weights[batch_idx, head_idx] = F.softmax(
                            attention_weights[batch_idx, head_idx], dim=-1
                        )
            
            return (hidden_states, attention_weights)
        
        return pattern_attention_intervention_hook
    
    def apply_pattern_intervention(self, target_layer, pattern_positions, intervention_strength=0.8):
        """Apply pattern-based attention intervention to specific layer."""
        
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
        hook = self.create_pattern_attention_hook(target_layer, pattern_positions, intervention_strength)
        handle = attention_module.register_forward_hook(hook)
        
        return handle
    
    def generate_with_pattern_intervention(self, input_ids, target_layer, emerging_patterns, 
                                         max_new_tokens=100, intervention_strength=0.8):
        """Generate text with pattern-based attention intervention."""
        
        # Extract pattern positions for intervention
        pattern_positions = []
        for batch_idx in range(input_ids.shape[0]):
            batch_positions = []
            for pattern, info in emerging_patterns.items():
                batch_positions.extend(info['positions'])
            pattern_positions.append(batch_positions)
        
        # Apply pattern intervention
        hook_handle = self.apply_pattern_intervention(
            target_layer, pattern_positions, intervention_strength
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
    
    def analyze_pattern_repetition_induction(self, texts, target_layer, intervention_strength=0.8, 
                                           max_new_tokens=100):
        """Test whether forcing attention to emerging patterns induces repetition."""
        
        results = {
            'baseline_generations': [],
            'intervention_generations': [],
            'baseline_cycles': [],
            'intervention_cycles': [],
            'repetition_induced': [],
            'detected_patterns': [],
            'input_texts': texts
        }
        
        print(f"🧪 Testing emerging pattern attention intervention on layer {target_layer}")
        print(f"💪 Intervention strength: {intervention_strength}")
        
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Detect emerging patterns in input
            tokens = input_ids[0].cpu()  # First (and only) batch item
            emerging_patterns = self.detect_emerging_patterns(tokens)
            results['detected_patterns'].append(emerging_patterns)
            
            # Skip if no patterns detected
            if not emerging_patterns:
                print(f"   ⚠️ No emerging patterns detected in text {i+1}, using baseline")
                # Just run baseline
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
                
                # Store as both baseline and intervention
                results['baseline_generations'].append(baseline_text)
                results['intervention_generations'].append(baseline_text)
                results['baseline_cycles'].append(baseline_cycles)
                results['intervention_cycles'].append(baseline_cycles)
                results['repetition_induced'].append(False)
                continue
            
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
            
            # 2. Pattern intervention generation
            intervention_output = self.generate_with_pattern_intervention(
                input_ids, target_layer, emerging_patterns, max_new_tokens, intervention_strength
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
                print(f"Patterns detected: {len(emerging_patterns)}")
                for pattern, info in list(emerging_patterns.items())[:3]:  # Show first 3 patterns
                    pattern_tokens = [self.tokenizer.decode([t]) for t in pattern]
                    print(f"  Pattern: {pattern_tokens} (freq: {info['frequency']})")
                print(f"Baseline cycles: {baseline_cycle_count}")
                print(f"Intervention cycles: {intervention_cycle_count}")
                print(f"Repetition induced: {repetition_induced}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Emerging Pattern Sequence Causal Intervention Experiment")
    parser.add_argument("--layer", type=int, default=15, help="Target layer for intervention")
    parser.add_argument("--strength", type=float, default=0.8, help="Intervention strength")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Emerging Pattern Sequence Causal Intervention Experiment")
    print(f"📋 Parameters:")
    print(f"   - Target layer: {args.layer}")
    print(f"   - Intervention strength: {args.strength}")
    print(f"   - Number of samples: {args.n_samples}")
    
    # Load model and tokenizer
    print(f"🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=args.checkpoint)
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Load dataset with pattern potential
    print(f"📚 Loading dataset...")
    try:
        dataset = load_text_dataset("JeanKaddour/minipile")
        test_texts = dataset[:args.n_samples]
    except:
        print("Failed to load JeanKaddour/minipile dataset, using wikitext...")
        dataset = load_text_dataset("wikitext")
        test_texts = dataset[:args.n_samples]
    
    print(f"📊 Loaded {len(test_texts)} texts for pattern intervention testing")
    
    # Create output directory
    output_dir = Path(f"./plots/pattern_intervention_L{args.layer}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize pattern intervention processor
    processor = EmergingPatternProcessor(model, tokenizer)
    
    # Run pattern causal intervention experiment
    print(f"🧪 Running emerging pattern causal intervention experiment...")
    results = processor.analyze_pattern_repetition_induction(
        test_texts, args.layer, args.strength
    )
    
    # Save raw results
    results_path = output_dir / f"pattern_intervention_results_L{args.layer}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Analyze pattern detection statistics
    total_patterns = sum(len(patterns) for patterns in results['detected_patterns'])
    texts_with_patterns = sum(1 for patterns in results['detected_patterns'] if patterns)
    
    # Create report
    induction_rate = np.mean(results['repetition_induced'])
    report_path = output_dir / f"pattern_intervention_report_L{args.layer}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Emerging Pattern Sequence Causal Intervention Report\\n\\n")
        f.write(f"**Target Layer**: {args.layer}  \\n")
        f.write(f"**Intervention Strength**: {args.strength}  \\n")
        f.write(f"**Total Texts Tested**: {len(test_texts)}  \\n\\n")
        
        f.write(f"## Pattern Detection Statistics\\n\\n")
        f.write(f"- **Total Patterns Detected**: {total_patterns}\\n")
        f.write(f"- **Texts with Patterns**: {texts_with_patterns}/{len(test_texts)} ({texts_with_patterns/len(test_texts):.1%})\\n")
        f.write(f"- **Average Patterns per Text**: {total_patterns/len(test_texts):.1f}\\n\\n")
        
        f.write(f"## Summary Results\\n\\n")
        f.write(f"- **Repetition Induction Rate**: {induction_rate:.2%} ({sum(results['repetition_induced'])}/{len(results['repetition_induced'])})\\n")
        
        if induction_rate > 0.15:
            f.write(f"\\n✅ **HYPOTHESIS SUPPORTED**: Pattern-based attention intervention induces repetition at {induction_rate:.1%} rate.\\n\\n")
            f.write(f"**Implication**: Forcing attention to emerging patterns can trigger repetitive generation.\\n\\n")
        else:
            f.write(f"\\n❌ **HYPOTHESIS NOT SUPPORTED**: Low induction rate ({induction_rate:.1%}) suggests pattern attention is insufficient.\\n\\n")
            f.write(f"**Implication**: Simple pattern attention manipulation does not reliably induce repetition.\\n\\n")
        
        f.write(f"## Mechanism Insights\\n\\n")
        f.write(f"- **Pattern sensitivity**: {'High' if texts_with_patterns/len(test_texts) > 0.7 else 'Moderate' if texts_with_patterns/len(test_texts) > 0.3 else 'Low'}\\n")
        f.write(f"- **Intervention effectiveness**: {'Strong' if induction_rate > 0.15 else 'Weak'}\\n")
        f.write(f"- **Causal pathway**: {'Pattern attention drives repetition' if induction_rate > 0.15 else 'Pattern attention insufficient for repetition induction'}\\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\\n🎯 Pattern Intervention Summary:")
    print(f"   - Repetition induction rate: {induction_rate:.2%}")
    print(f"   - Successful interventions: {sum(results['repetition_induced'])}/{len(results['repetition_induced'])}")
    print(f"   - Patterns detected: {total_patterns} across {texts_with_patterns} texts")
    
    if induction_rate > 0.15:
        print(f"   ✅ PATTERN EFFECT: Emerging pattern attention induces repetition")
    else:
        print(f"   ❌ NO PATTERN EFFECT: Pattern intervention ineffective")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()