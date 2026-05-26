#!/usr/bin/env python3
"""
Quick Newline Causality Test - Direct and Simple
"""

import torch
import sys
import os
from datetime import datetime

# Add paths for imports  
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu/modules')

try:
    from model_utils import load_model_and_tokenizer
    print("✅ Imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)


def count_repetitions(text):
    """Simple repetition counter."""
    words = text.lower().split()
    if len(words) < 4:
        return 0
    
    repetitions = 0
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repetitions += 1
        
        # Check for 2-word repeats
        if i < len(words) - 3:
            if words[i:i+2] == words[i+2:i+4]:
                repetitions += 2
    
    return repetitions


def main():
    print("🧪 QUICK NEWLINE CAUSALITY TEST")
    print("=" * 50)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Get newline token
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    print(f"✅ Newline token ID: {newline_token_id}")
    
    # Test prompts with and without newlines
    test_cases = [
        {
            'name': 'List with newlines',
            'with_newlines': "Items:\nApple\nBanana\nCherry\nDate\n",
            'without_newlines': "Items: Apple Banana Cherry Date "
        },
        {
            'name': 'Steps with newlines', 
            'with_newlines': "Process:\n1. Start\n2. Continue\n3. Finish\n",
            'without_newlines': "Process: 1. Start 2. Continue 3. Finish "
        },
        {
            'name': 'Paragraphs with newlines',
            'with_newlines': "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n",
            'without_newlines': "First paragraph. Second paragraph. Third paragraph. "
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n--- Testing: {case['name']} ---")
        
        # Generate with newlines
        input_ids_nl = tokenizer.encode(case['with_newlines'], return_tensors="pt").to(device)
        with torch.no_grad():
            output_nl = model.generate(
                input_ids_nl,
                max_length=200,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        text_nl = tokenizer.decode(output_nl[0], skip_special_tokens=True)
        generated_nl = text_nl[len(case['with_newlines']):]
        reps_nl = count_repetitions(generated_nl)
        
        # Generate without newlines
        input_ids_no_nl = tokenizer.encode(case['without_newlines'], return_tensors="pt").to(device)
        with torch.no_grad():
            output_no_nl = model.generate(
                input_ids_no_nl,
                max_length=200,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        text_no_nl = tokenizer.decode(output_no_nl[0], skip_special_tokens=True)
        generated_no_nl = text_no_nl[len(case['without_newlines']):]
        reps_no_nl = count_repetitions(generated_no_nl)
        
        # Compare
        effect = "CAUSAL" if reps_no_nl < reps_nl else "NO_EFFECT"
        reduction = reps_nl - reps_no_nl
        
        result = {
            'case': case['name'],
            'with_newlines_reps': reps_nl,
            'without_newlines_reps': reps_no_nl,
            'reduction': reduction,
            'effect': effect,
            'sample_with_nl': generated_nl[:100],
            'sample_without_nl': generated_no_nl[:100]
        }
        results.append(result)
        
        print(f"  With newlines: {reps_nl} repetitions")
        print(f"  Without newlines: {reps_no_nl} repetitions")
        print(f"  Effect: {effect} (reduction: {reduction})")
        print(f"  Sample with NL: {generated_nl[:80]}...")
        print(f"  Sample no NL: {generated_no_nl[:80]}...")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CAUSALITY TEST SUMMARY")
    print("=" * 50)
    
    causal_cases = sum(1 for r in results if r['effect'] == 'CAUSAL')
    total_reduction = sum(r['reduction'] for r in results)
    
    print(f"Cases showing causal effect: {causal_cases}/{len(results)}")
    print(f"Total repetition reduction: {total_reduction}")
    
    if causal_cases >= 2:
        print("🎯 CONCLUSION: Strong evidence for newline causality!")
        print("🎉 Newlines appear to causally drive repetition behavior!")
    elif causal_cases == 1:
        print("📈 CONCLUSION: Moderate evidence for newline causality")
        print("🔍 Some causal effect detected, worth further investigation")
    else:
        print("❌ CONCLUSION: No clear causal evidence")
        print("🔄 Newlines may be correlational, not causal")
    
    # Save results
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    output_file = os.path.join(output_dir, 'quick_newline_causality.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'newline_token_id': newline_token_id,
            'results': results,
            'summary': {
                'causal_cases': causal_cases,
                'total_cases': len(results),
                'total_reduction': total_reduction,
                'evidence_strength': causal_cases / len(results) * 100
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return causal_cases >= 1


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)