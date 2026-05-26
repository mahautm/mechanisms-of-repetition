#!/usr/bin/env python3
"""
Final Validation: Confirmed Repetition Induction Techniques

Based on our breakthrough discovery, this validates our most effective
repetition induction techniques for reliable deployment.
"""

import torch
import sys
import os
import json
from datetime import datetime

# Add paths for imports
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu/modules')

try:
    from model_utils import load_model_and_tokenizer
    from parrots.cycle_detection import detect_cycles
    print("✅ Imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)


def count_cycles_standardized(text, tokenizer):
    """Standardized cycle counting for validation using token-based detect_cycles."""
    if not text or not text.strip():
        return {"total": 0, "cycle_count": 0, "cycle_size": 0}
    
    try:
        # Tokenize the text
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        
        # Use standard detect_cycles function
        cycle, cycle_size, cycle_count = detect_cycles(tokens.tolist())
        
        return {
            "total": cycle_count if cycle_count else 0,
            "cycle_count": cycle_count if cycle_count else 0,
            "cycle_size": cycle_size
        }
    except Exception as e:
        print(f"Warning: Error in standardized cycle counting: {e}")
        return {"total": 0, "cycle_count": 0, "cycle_size": 0}


    if len(words) < 2:
        return {'total': 0, 'immediate': 0, 'phrase': 0, 'long_runs': 0}
    
    immediate = 0
    phrase = 0
    long_runs = 0
    
    # Immediate repetition (word word)
    current_run = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            current_run += 1
            immediate += 1
        else:
            if current_run >= 5:  # Long runs of 5+ same words
                long_runs += 1
            current_run = 1
    
    if current_run >= 5:
        long_runs += 1
    
    # Phrase repetition (2-4 word sequences)
    for length in [2, 3, 4]:
        for i in range(len(words) - length * 2):
            if words[i:i+length] == words[i+length:i+length*2]:
                phrase += 1
    
    return {
        'total': immediate + phrase + long_runs,
        'immediate': immediate,
        'phrase': phrase,
        'long_runs': long_runs
    }


def main():
    print("🎯 FINAL VALIDATION: CONFIRMED REPETITION TECHNIQUES")
    print("=" * 65)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # Our confirmed most effective techniques from discovery phase
    proven_techniques = {
        'ultra_high': [
            "again and again and",
            "the thing that contains the thing that contains"
        ],
        'high': [
            "over and over and",
            "more and more and",
            "round and round and"
        ],
        'recursive_confirmed': [
            "people who know people who know",
            "systems that control systems that control",
            "processes that create processes that create"
        ],
        'conversational_confirmed': [
            "he said she said he said she said",
            "they told us we told them they told us",
            "A asked B who asked A who asked B"
        ],
        'optimized_combinations': [
            "again and again the pattern that contains the pattern",
            "over and over the cycle within the cycle within",
            "more and more they said we said they said"
        ]
    }
    
    # Control baselines
    control_prompts = [
        "The weather is nice today and",
        "Scientists have recently discovered that",
        "In the field of computer science",
        "The quick brown fox jumped over"
    ]
    
    all_results = {}
    
    # Test control baselines first
    print("\n--- Testing Control Baselines ---")
    control_results = []
    for prompt in control_prompts:
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
        reps = count_cycles_standardized(generated_only, tokenizer)
        control_results.append(reps['total'])
        
        print(f"  '{prompt[:30]}...' → {reps['total']} reps")
    
    baseline_avg = sum(control_results) / len(control_results)
    print(f"  → Baseline Average: {baseline_avg:.1f} repetitions")
    
    all_results['baseline'] = {
        'results': control_results,
        'average': baseline_avg
    }
    
    # Test our proven techniques
    for category, techniques in proven_techniques.items():
        print(f"\n--- Testing {category.replace('_', ' ').title()} Techniques ---")
        
        category_results = []
        
        for technique in techniques:
            input_ids = tokenizer.encode(technique, return_tensors="pt").to(device)
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
            generated_only = generated_text[len(technique):]
            
            reps = count_cycles_standardized(generated_only, tokenizer)
            total_reps = reps['total']
            category_results.append(total_reps)
            
            # Show effectiveness vs baseline
            effectiveness = (total_reps / (baseline_avg + 0.1))  # Avoid division by zero
            
            print(f"  '{technique[:35]}...' → {total_reps} reps ({effectiveness:.1f}x baseline)")
            
            # Show sample of generated text if high repetition
            if total_reps > 5:
                print(f"    Sample: '{generated_only[:80]}...'")
        
        category_avg = sum(category_results) / len(category_results) if category_results else 0
        category_max = max(category_results) if category_results else 0
        
        print(f"  → Category Average: {category_avg:.1f} reps (Max: {category_max})")
        
        all_results[category] = {
            'results': category_results,
            'average': category_avg,
            'max': category_max,
            'effectiveness': category_avg / (baseline_avg + 0.1)
        }
    
    # Final Analysis
    print("\n" + "=" * 65)
    print("📊 FINAL VALIDATION ANALYSIS")
    print("=" * 65)
    
    # Rank categories by effectiveness
    effectiveness_ranking = []
    for category, data in all_results.items():
        if category != 'baseline':
            effectiveness_ranking.append((category, data['average'], data['max'], data['effectiveness']))
    
    effectiveness_ranking.sort(key=lambda x: x[1], reverse=True)  # Sort by average
    
    print(f"\nBaseline Average: {baseline_avg:.1f} repetitions")
    print(f"\nTechnique Effectiveness Ranking:")
    
    deployment_ready = []
    
    for i, (category, avg, max_reps, effectiveness) in enumerate(effectiveness_ranking, 1):
        status = "🎯 DEPLOYMENT READY" if effectiveness >= 5.0 else "📈 EFFECTIVE" if effectiveness >= 2.0 else "❌ LIMITED"
        
        print(f"  {i}. {category.replace('_', ' ').title()}:")
        print(f"     Average: {avg:.1f} reps | Max: {max_reps} | Effectiveness: {effectiveness:.1f}x | {status}")
        
        if effectiveness >= 5.0:
            deployment_ready.append(category)
    
    # Overall conclusion
    best_technique = effectiveness_ranking[0] if effectiveness_ranking else ('none', 0, 0, 0)
    max_effectiveness = best_technique[3]
    
    print(f"\n🎯 VALIDATION CONCLUSION:")
    
    if max_effectiveness >= 10.0:
        conclusion = "🎉 BREAKTHROUGH CONFIRMED: Ultra-high effectiveness achieved!"
        status = "READY FOR IMMEDIATE DEPLOYMENT"
    elif max_effectiveness >= 5.0:
        conclusion = "✅ SUCCESS VALIDATED: High effectiveness confirmed!"
        status = "READY FOR DEPLOYMENT"
    elif max_effectiveness >= 2.0:
        conclusion = "📈 MODERATE SUCCESS: Techniques show clear effectiveness!"
        status = "REFINEMENT RECOMMENDED"
    else:
        conclusion = "❌ LIMITED EFFECTIVENESS: Further development needed"
        status = "NOT READY FOR DEPLOYMENT"
    
    print(f"  {conclusion}")
    print(f"  Status: {status}")
    print(f"  Best Technique: {best_technique[0].replace('_', ' ').title()} ({max_effectiveness:.1f}x baseline)")
    print(f"  Deployment-Ready Techniques: {len(deployment_ready)}")
    
    # Save validation results
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    validation_summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_average': baseline_avg,
        'technique_rankings': effectiveness_ranking,
        'best_technique': best_technique[0],
        'max_effectiveness': max_effectiveness,
        'deployment_ready_count': len(deployment_ready),
        'deployment_ready_techniques': deployment_ready,
        'conclusion': conclusion,
        'status': status,
        'validation_successful': max_effectiveness >= 2.0
    }
    
    output_file = os.path.join(output_dir, 'final_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(validation_summary, f, indent=2)
    
    print(f"\n💾 Validation results saved to: {output_file}")
    
    # Create deployment recommendations if successful
    if max_effectiveness >= 5.0:
        print(f"\n🚀 DEPLOYMENT RECOMMENDATIONS:")
        print(f"  1. Use '{best_technique[0].replace('_', ' ')}' techniques for maximum effect")
        print(f"  2. Expected effectiveness: {max_effectiveness:.1f}x baseline repetition")
        print(f"  3. Suitable for: Controlled repetition generation research")
        print(f"  4. Safety note: Monitor for excessive repetition in production use")
    
    return max_effectiveness >= 2.0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)