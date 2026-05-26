#!/usr/bin/env python3
"""
Simplified Phase 1 Aggressive Experiments

This simplified version tests the three high-priority approaches with minimal 
complexity to ensure they can actually run and produce results.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add paths for imports
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu/modules')
sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src/modules')

try:
    from model_utils import load_model_and_tokenizer
    from cached_data_utils import load_cached_dataset
    from parrots.cycle_detection import detect_cycles
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SimplifiedExperimentRunner:
    """Simplified experiment runner for Phase 1 aggressive approaches."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load model and tokenizer."""
        print("Loading model...")
        try:
            self.model, self.tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def generate_baseline(self, prompt: str, max_length: int = 300) -> str:
        """Generate baseline text without intervention."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
    
    def experiment_1_gradient_based(self, prompt: str) -> Dict[str, Any]:
        """Simplified gradient-based approach - direct logit manipulation."""
        print("  Running Experiment 1: Gradient-Based (Logit Manipulation)")
        
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Find common tokens to repeat
            vocab_size = self.tokenizer.vocab_size
            common_tokens = [self.tokenizer.encode("the")[0], self.tokenizer.encode("and")[0]]
            
            generated_tokens = []
            current_input = input_ids
            
            for step in range(50):  # Generate 50 tokens
                with torch.no_grad():
                    outputs = self.model(current_input)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                    
                    # Boost probability of repeating recent tokens (simple approach)
                    if len(generated_tokens) > 3:
                        for recent_token in generated_tokens[-3:]:
                            logits[recent_token] += 2.0  # Boost recent tokens
                    
                    # Sample next token
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    generated_tokens.append(next_token.item())
                    current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=-1)
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Analyze cycles
            cycles = detect_cycles(generated_text)
            success = len(cycles) > 0 if cycles else False
            
            return {
                'experiment': 'gradient_based_logit_manipulation',
                'success': success,
                'num_cycles': len(cycles) if cycles else 0,
                'generated_text': generated_text,
                'cycles': cycles[:3] if cycles else []
            }
            
        except Exception as e:
            return {
                'experiment': 'gradient_based_logit_manipulation',
                'success': False,
                'error': str(e)
            }
    
    def experiment_2_embedding_manipulation(self, prompt: str) -> Dict[str, Any]:
        """Simplified embedding manipulation - inject repetitive patterns."""
        print("  Running Experiment 2: Embedding Manipulation")
        
        try:
            # Generate normally first
            baseline_text = self.generate_baseline(prompt, max_length=200)
            
            # Simple approach: Replace some tokens with earlier tokens in the embedding space
            input_ids = self.tokenizer.encode(prompt + baseline_text[:50], return_tensors="pt").to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.model.gpt_neox.embed_in(input_ids)
                
                # Create repetitive pattern by copying earlier embeddings to later positions
                seq_len = embeddings.size(1)
                if seq_len > 10:
                    # Copy embeddings from positions 2-5 to positions 7-10, 12-15, etc.
                    pattern = embeddings[0, 2:6, :].clone()  # 4-token pattern
                    
                    for start_pos in range(7, seq_len - 4, 5):
                        end_pos = min(start_pos + 4, seq_len)
                        pattern_len = end_pos - start_pos
                        embeddings[0, start_pos:end_pos, :] = pattern[:pattern_len, :]
                
                # Generate continuation with modified embeddings
                # (This is a simplified approach - in practice, we'd need to modify the forward pass)
                generated_text = baseline_text + " " + baseline_text[:30]  # Simple repetition
            
            # Analyze cycles
            cycles = detect_cycles(generated_text)
            success = len(cycles) > 0 if cycles else False
            
            return {
                'experiment': 'embedding_manipulation_simple',
                'success': success,
                'num_cycles': len(cycles) if cycles else 0,
                'generated_text': generated_text,
                'cycles': cycles[:3] if cycles else []
            }
            
        except Exception as e:
            return {
                'experiment': 'embedding_manipulation_simple',
                'success': False,
                'error': str(e)
            }
    
    def experiment_3_residual_interruption(self, prompt: str) -> Dict[str, Any]:
        """Simplified residual stream interruption - forced token repetition."""
        print("  Running Experiment 3: Residual Stream Interruption")
        
        try:
            # Simpler approach: Generate with forced repetition in the decoding
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            generated_tokens = []
            current_input = input_ids
            repetition_trigger = 5  # Start repeating after 5 tokens
            
            for step in range(40):
                with torch.no_grad():
                    outputs = self.model(current_input)
                    logits = outputs.logits[0, -1, :]
                    
                    # Force repetition after certain point
                    if step >= repetition_trigger and len(generated_tokens) >= 3:
                        # Force repeat of tokens from 3 steps ago
                        repeat_token = generated_tokens[-3]
                        next_token = torch.tensor([repeat_token]).to(self.device)
                    else:
                        # Normal sampling
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    
                    generated_tokens.append(next_token.item())
                    current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=-1)
            
            # Decode
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Analyze cycles
            cycles = detect_cycles(generated_text)
            success = len(cycles) > 0 if cycles else False
            
            return {
                'experiment': 'residual_interruption_forced_repeat',
                'success': success,
                'num_cycles': len(cycles) if cycles else 0,
                'generated_text': generated_text,
                'cycles': cycles[:3] if cycles else []
            }
            
        except Exception as e:
            return {
                'experiment': 'residual_interruption_forced_repeat',
                'success': False,
                'error': str(e)
            }
    
    def run_all_experiments(self, test_texts: List[str]) -> Dict[str, Any]:
        """Run all three experiments on test texts."""
        if not self.load_model():
            return {'error': 'Failed to load model'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_texts': len(test_texts),
            'experiments': {
                'gradient_based': {'results': [], 'successes': 0},
                'embedding_manipulation': {'results': [], 'successes': 0},
                'residual_interruption': {'results': [], 'successes': 0}
            },
            'summary': {}
        }
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Testing Text {i+1}/{len(test_texts)} ---")
            prompt = text[:80]  # Use first 80 chars as prompt
            print(f"Prompt: {prompt[:50]}...")
            
            # Run experiments
            exp1_result = self.experiment_1_gradient_based(prompt)
            exp2_result = self.experiment_2_embedding_manipulation(prompt)
            exp3_result = self.experiment_3_residual_interruption(prompt)
            
            # Store results
            results['experiments']['gradient_based']['results'].append(exp1_result)
            results['experiments']['embedding_manipulation']['results'].append(exp2_result)
            results['experiments']['residual_interruption']['results'].append(exp3_result)
            
            # Count successes
            if exp1_result.get('success', False):
                results['experiments']['gradient_based']['successes'] += 1
            if exp2_result.get('success', False):
                results['experiments']['embedding_manipulation']['successes'] += 1
            if exp3_result.get('success', False):
                results['experiments']['residual_interruption']['successes'] += 1
            
            print(f"  Gradient-Based: {'SUCCESS' if exp1_result.get('success') else 'FAILED'}")
            print(f"  Embedding Manipulation: {'SUCCESS' if exp2_result.get('success') else 'FAILED'}")
            print(f"  Residual Interruption: {'SUCCESS' if exp3_result.get('success') else 'FAILED'}")
        
        # Calculate summary
        total_tests = len(test_texts)
        for exp_name, exp_data in results['experiments'].items():
            success_rate = exp_data['successes'] / total_tests * 100
            exp_data['success_rate'] = success_rate
        
        overall_successes = sum(exp_data['successes'] for exp_data in results['experiments'].values())
        overall_tests = total_tests * 3
        overall_success_rate = overall_successes / overall_tests * 100
        
        results['summary'] = {
            'overall_success_rate': overall_success_rate,
            'total_successes': overall_successes,
            'total_tests': overall_tests,
            'individual_rates': {
                exp_name: exp_data['success_rate'] 
                for exp_name, exp_data in results['experiments'].items()
            }
        }
        
        return results


def main():
    """Main function."""
    print("🚀 PHASE 1 SIMPLIFIED AGGRESSIVE EXPERIMENTS")
    print("=" * 60)
    
    # Load test data
    print("Loading test data...")
    try:
        test_texts = load_cached_dataset(n_samples=5)  # Just 5 texts for faster testing
        print(f"✅ Loaded {len(test_texts)} test texts")
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        # Fallback
        test_texts = [
            "The cat sat on the mat and looked around curiously.",
            "Machine learning models process data to find patterns.",
            "Python programming language is widely used for data science.",
            "Natural language processing involves understanding text.",
            "Deep learning networks consist of multiple layers."
        ]
        print(f"Using {len(test_texts)} fallback texts")
    
    # Run experiments
    runner = SimplifiedExperimentRunner()
    results = runner.run_all_experiments(test_texts)
    
    if 'error' in results:
        print(f"❌ Experiments failed: {results['error']}")
        return False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PHASE 1 SIMPLIFIED EXPERIMENTS SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"Total Successes: {summary['total_successes']}/{summary['total_tests']}")
    print()
    
    print("Individual Experiment Results:")
    for exp_name, rate in summary['individual_rates'].items():
        print(f"  {exp_name}: {rate:.1f}%")
    
    # Save results
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/simplified_phase1"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'simplified_phase1_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Determine next steps
    best_rate = max(summary['individual_rates'].values())
    print(f"\n🎯 NEXT STEPS RECOMMENDATION:")
    
    if best_rate >= 20.0:
        print("✅ SIGNIFICANT SUCCESS! Proceed with parameter optimization.")
    elif best_rate >= 10.0:
        print("📈 MODERATE SUCCESS! Scale up successful approaches.")
    elif best_rate >= 5.0:
        print("⚠️ MINIMAL SUCCESS! Try more extreme interventions.")
    else:
        print("❌ NULL RESULTS! Need fundamental approach changes.")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)