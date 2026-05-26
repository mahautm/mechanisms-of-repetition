#!/usr/bin/env python3
"""
Quick Newline Causal Test

Based on the discovery that newline token ID = 187, test direct causal interventions
to determine if newlines are causing repetition.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu/modules')
sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src/modules')

from model_utils import load_model_and_tokenizer
from cached_data_utils import load_cached_dataset
from parrots.cycle_detection import detect_cycles
import torch

def test_newline_causality():
    """Test if newlines causally drive repetition."""
    print("🔬 NEWLINE CAUSALITY TEST")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    model = model.to("cuda")
    model.eval()
    
    # Verify newline token
    newline_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    print(f"✅ Newline token ID confirmed: {newline_id}")
    
    # Test texts with different newline patterns
    test_cases = [
        {
            'name': 'Original with newlines',
            'text': "This is line 1.\nThis is line 2.\nThis is line 3.\nRepeat pattern here."
        },
        {
            'name': 'No newlines (spaces)',
            'text': "This is line 1. This is line 2. This is line 3. Repeat pattern here."
        },
        {
            'name': 'Double newlines',
            'text': "This is line 1.\n\nThis is line 2.\n\nThis is line 3.\n\nRepeat pattern here."
        },
        {
            'name': 'Replaced with periods',
            'text': "This is line 1. . This is line 2. . This is line 3. . Repeat pattern here."
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n--- Testing: {case['name']} ---")
        
        # Generate text
        input_ids = tokenizer.encode(case['text'][:80], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=200,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_part = generated[len(case['text'][:80]):]
        
        # Detect cycles
        cycles = detect_cycles(generated_part)
        cycle_count = len(cycles) if cycles else 0
        
        # Count newlines in original prompt
        newline_count = case['text'].count('\n')
        
        result = {
            'case': case['name'],
            'newline_count': newline_count,
            'cycle_count': cycle_count,
            'generated_sample': generated_part[:100]
        }
        results.append(result)
        
        print(f"  Newlines in prompt: {newline_count}")
        print(f"  Cycles detected: {cycle_count}")
        print(f"  Sample: {generated_part[:50]}...")
    
    # Analysis
    print(f"\n" + "=" * 50)
    print("📊 CAUSALITY ANALYSIS")
    print("=" * 50)
    
    baseline_cycles = next((r['cycle_count'] for r in results if r['case'] == 'Original with newlines'), 0)
    no_newline_cycles = next((r['cycle_count'] for r in results if r['case'] == 'No newlines (spaces)'), 0)
    double_newline_cycles = next((r['cycle_count'] for r in results if r['case'] == 'Double newlines'), 0)
    
    print(f"Original (with newlines): {baseline_cycles} cycles")
    print(f"No newlines: {no_newline_cycles} cycles")  
    print(f"Double newlines: {double_newline_cycles} cycles")
    
    # Causal evidence
    evidence = []
    if no_newline_cycles < baseline_cycles:
        evidence.append("✅ REMOVING newlines REDUCES repetition")
    if double_newline_cycles > baseline_cycles:
        evidence.append("✅ ADDING newlines INCREASES repetition")
    
    if evidence:
        print(f"\n⚡ CAUSAL EVIDENCE FOUND:")
        for e in evidence:
            print(f"  {e}")
        print(f"\n🎯 CONCLUSION: Newlines appear to CAUSE repetition!")
    else:
        print(f"\n❌ No clear causal evidence detected")
        print(f"🤔 Newlines may be correlated but not causal")
    
    return results

if __name__ == "__main__":
    test_newline_causality()