#!/usr/bin/env python3
"""
Quick Causal Mechanism Explorer - Fast Tests for Immediate Insights
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
    from parrots.cycle_detection import detect_cycles
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    exit(1)


def count_cycles_standardized(text, tokenizer):
    """Standardized cycle detection using token-based approach."""
    if not text or not text.strip():
        return {'cycle_count': 0, 'cycle_size': 0, 'total': 0}
    
    try:
        # Tokenize the text
        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        
        # Use standard detect_cycles function
        cycle, cycle_size, cycle_count = detect_cycles(tokens.tolist())
        
        return {
            'cycle_count': cycle_count if cycle_count else 0,
            'cycle_size': cycle_size,
            'total': cycle_count if cycle_count else 0
        }
    except Exception as e:
        print(f"Warning: Error in cycle detection: {e}")
        return {'cycle_count': 0, 'cycle_size': 0, 'total': 0}


def main():
    print("🔬 QUICK CAUSAL MECHANISM EXPLORATION")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # Test categories with different hypothesis
    test_categories = {
        'repetitive_words': [
            "again and again and",
            "more and more and",
            "over and over and",
            "round and round and"
        ],
        'listing_patterns': [
            "first, second, third,",
            "item A, item B, item C,",
            "step 1, step 2, step 3,",
            "point one, point two, point three,"
        ],
        'recursive_structures': [
            "the thing that contains the thing that contains",
            "people who know people who know",
            "systems that control systems that control",
            "processes that create processes that create"
        ],
        'incomplete_patterns': [
            "if this then that, if that then",
            "not only A but also B, not only B but also",
            "either X or Y, either Y or",
            "both P and Q, both Q and"
        ],
        'training_artifacts': [
            "see more at example.com, see more at",
            "copyright 2023, all rights reserved, copyright 2023,",
            "for more information please visit, for more information please",
            "click here to continue, click here to"
        ],
        'neutral_baseline': [
            "the quick brown fox jumped over",
            "in a small town near the mountains",
            "scientists have recently discovered that",
            "the weather forecast shows partly cloudy"
        ]
    }
    
    results = {}
    
    for category_name, prompts in test_categories.items():
        print(f"\n--- Testing {category_name.replace('_', ' ').title()} ---")
        
        category_results = []
        
        for prompt in prompts:
            # Generate text
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=min(len(input_ids[0]) + 150, 250),
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    repetition_penalty=1.0
                )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_only = generated_text[len(prompt):]
            
            # Count repetitions
            rep_counts = count_cycles_standardized(generated_only, tokenizer)
            
            result = {
                'prompt': prompt,
                'generated': generated_only[:100],  # First 100 chars
                'repetitions': rep_counts
            }
            category_results.append(result)
            
            print(f"  '{prompt[:30]}...' → {rep_counts['total']} reps")
        
        # Calculate category average
        avg_reps = sum(r['repetitions']['total'] for r in category_results) / len(category_results)
        results[category_name] = {
            'results': category_results,
            'avg_repetitions': avg_reps,
            'max_repetitions': max(r['repetitions']['total'] for r in category_results)
        }
        
        print(f"  → Average: {avg_reps:.1f} repetitions")
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 CAUSAL MECHANISM ANALYSIS")
    print("=" * 60)
    
    # Rank categories by effectiveness
    ranked_categories = sorted(results.items(), 
                              key=lambda x: x[1]['avg_repetitions'], 
                              reverse=True)
    
    print("\nCausal Effectiveness Ranking:")
    for i, (category, data) in enumerate(ranked_categories, 1):
        print(f"  {i}. {category.replace('_', ' ').title()}: {data['avg_repetitions']:.1f} avg reps "
              f"(max: {data['max_repetitions']})")
    
    # Find most promising mechanism
    most_effective = ranked_categories[0]
    baseline_avg = results['neutral_baseline']['avg_repetitions']
    
    effectiveness_ratio = most_effective[1]['avg_repetitions'] / (baseline_avg + 0.1)  # Avoid division by zero
    
    print(f"\nMost Effective Mechanism: {most_effective[0].replace('_', ' ').title()}")
    print(f"Effectiveness vs Baseline: {effectiveness_ratio:.1f}x")
    
    if effectiveness_ratio >= 3.0:
        conclusion = "🎯 STRONG CAUSAL EVIDENCE FOUND!"
        next_action = f"Intensive investigation of {most_effective[0]} mechanisms"
    elif effectiveness_ratio >= 1.5:
        conclusion = "📈 MODERATE CAUSAL EVIDENCE"
        next_action = f"Deeper testing of {most_effective[0]} patterns"
    else:
        conclusion = "❌ WEAK CAUSAL EVIDENCE"
        next_action = "Continue broader mechanism search"
    
    print(f"\n{conclusion}")
    print(f"Next Action: {next_action}")
    
    # Show top examples
    if most_effective[1]['avg_repetitions'] > 0:
        print(f"\nTop Repetition Examples from {most_effective[0]}:")
        top_examples = sorted(most_effective[1]['results'], 
                            key=lambda x: x['repetitions']['total'], 
                            reverse=True)[:3]
        
        for i, example in enumerate(top_examples, 1):
            print(f"  {i}. Prompt: '{example['prompt']}'")
            print(f"     Generated: '{example['generated'][:80]}...'")
            print(f"     Repetitions: {example['repetitions']['total']}")
    
    # Save results
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    output_file = os.path.join(output_dir, 'quick_mechanism_exploration.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'analysis': {
                'ranked_categories': [(cat, data['avg_repetitions']) for cat, data in ranked_categories],
                'most_effective_mechanism': most_effective[0],
                'effectiveness_ratio': effectiveness_ratio,
                'conclusion': conclusion,
                'baseline_avg': baseline_avg
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return effectiveness_ratio >= 1.5  # Return success if moderate or strong evidence


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)