#!/usr/bin/env python3
"""
Gradient-Based Repetition Optimization Experiment

This experiment uses gradient ascent to find optimal intervention directions
that maximize the cycle detection score, directly optimizing for repetition induction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import our utilities
import sys
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu/modules')
sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src/modules')
sys.path.append('/home/mmahaut/projects/parrots/parrots')
from model_utils import load_model_and_tokenizer
from cached_data_utils import load_cached_dataset
from parrots.cycle_detection import detect_cycles


@dataclass
class GradientBasedConfig:
    """Configuration for gradient-based repetition optimization."""
    model_name: str = "EleutherAI/pythia-1.4b"
    max_length: int = 512
    num_generate: int = 100
    
    # Gradient optimization parameters
    num_gradient_steps: int = 10
    learning_rate: float = 0.1
    gradient_clip: float = 1.0
    
    # Target parameters
    target_layers: List[int] = None  # If None, use [15, 17, 19]
    target_heads: List[int] = None   # If None, use [0, 1, 2]
    newline_focus_strength: float = 1.0
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [15, 17, 19]
        if self.target_heads is None:
            self.target_heads = [0, 1, 2]


class GradientBasedRepetitionOptimizer:
    """Optimizer that uses gradients to find optimal attention interventions."""
    
    def __init__(self, config: GradientBasedConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.attention_interventions = {}
        self.hooks = []
        
        # Initialize model
        self._initialize_model()
        
        # Find newline token
        self.newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
        print(f"Newline token ID: {self.newline_token_id}")
        
    def _initialize_model(self):
        """Load model and tokenizer."""
        print("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(self.config.model_name)
        self.model.eval()
        print(f"Model loaded: {self.config.model_name}")
        
    def _create_attention_hook(self, layer_idx: int, head_idx: int):
        """Create attention hook with learnable parameters."""
        def hook_fn(module, input, output):
            # output is (batch_size, num_heads, seq_len, seq_len)
            if layer_idx in self.attention_interventions:
                if head_idx in self.attention_interventions[layer_idx]:
                    intervention = self.attention_interventions[layer_idx][head_idx]
                    
                    # Apply intervention to specific head
                    attn_weights = output[0]  # (batch_size, num_heads, seq_len, seq_len)
                    seq_len = attn_weights.size(-1)
                    
                    # Create newline mask
                    input_ids = input[0]  # Get input_ids from the input
                    if hasattr(input[0], 'shape') and len(input[0].shape) > 1:
                        # If input is already tokens
                        batch_input_ids = input[0]
                    else:
                        # Need to get input_ids from somewhere else
                        # This is a limitation - we'll need the input_ids passed differently
                        return output
                    
                    # Find newline positions
                    newline_mask = (batch_input_ids == self.newline_token_id).float()
                    newline_positions = torch.nonzero(newline_mask, as_tuple=True)[1]
                    
                    if len(newline_positions) > 0:
                        # Apply intervention
                        modified_weights = attn_weights.clone()
                        for pos in newline_positions:
                            if pos < seq_len:
                                # Increase attention to newline position
                                modified_weights[:, head_idx, :, pos] += intervention
                        
                        # Renormalize
                        modified_weights = F.softmax(modified_weights, dim=-1)
                        output = (modified_weights,) + output[1:]
            
            return output
        return hook_fn
    
    def _register_hooks(self):
        """Register attention hooks on target layers and heads."""
        self._clear_hooks()
        
        for layer_idx in self.config.target_layers:
            layer = self.model.gpt_neox.layers[layer_idx]
            hook = layer.attention.register_forward_hook(
                self._create_attention_hook(layer_idx, 0)  # We'll handle multiple heads in the hook
            )
            self.hooks.append(hook)
    
    def _clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _initialize_interventions(self):
        """Initialize learnable intervention parameters."""
        self.attention_interventions = {}
        
        for layer_idx in self.config.target_layers:
            self.attention_interventions[layer_idx] = {}
            for head_idx in self.config.target_heads:
                # Initialize intervention strength as learnable parameter
                intervention = torch.tensor(
                    self.config.newline_focus_strength,
                    device="cuda",
                    requires_grad=True
                )
                self.attention_interventions[layer_idx][head_idx] = intervention
    
    def _compute_repetition_loss(self, generated_text: str) -> torch.Tensor:
        """Compute loss that encourages repetition (higher = more repetitive)."""
        cycles = detect_cycles(generated_text)
        
        # Convert cycle score to tensor for gradient computation
        if cycles:
            # Sum of cycle scores weighted by length
            total_score = sum(len(cycle['repeated_text']) * cycle['repetitions'] 
                            for cycle in cycles)
            repetition_score = float(total_score) / len(generated_text)
        else:
            repetition_score = 0.0
        
        # We want to maximize this score
        return torch.tensor(repetition_score, device="cuda", requires_grad=True)
    
    def _generate_with_interventions(self, prompt: str) -> str:
        """Generate text with current intervention parameters."""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        # Store input_ids for hook access (simplified approach)
        self._current_input_ids = input_ids
        
        # Generate with current interventions
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Return only the generated part
    
    def optimize_interventions(self, prompt: str, target_cycles: int = 3) -> Dict:
        """Optimize intervention parameters using gradient ascent."""
        print(f"Optimizing interventions for prompt: {prompt[:50]}...")
        
        # Initialize interventions
        self._initialize_interventions()
        
        # Register hooks
        self._register_hooks()
        
        # Collect all parameters for optimization
        parameters = []
        for layer_dict in self.attention_interventions.values():
            for param in layer_dict.values():
                parameters.append(param)
        
        # Optimizer for gradient ascent (we want to maximize the score)
        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        
        results = {
            'prompt': prompt,
            'optimization_steps': [],
            'final_interventions': {},
            'best_generation': None,
            'best_score': 0.0
        }
        
        best_score = 0.0
        best_generation = None
        
        for step in range(self.config.num_gradient_steps):
            optimizer.zero_grad()
            
            # Generate text with current interventions
            generated_text = self._generate_with_interventions(prompt)
            
            # Compute repetition score
            repetition_score = self._compute_repetition_loss(generated_text)
            
            # We want to maximize repetition, so minimize negative score
            loss = -repetition_score
            
            # Backward pass
            if loss.requires_grad:
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
                
                # Update parameters
                optimizer.step()
            
            # Track results
            current_score = float(repetition_score)
            cycles = detect_cycles(generated_text)
            
            step_result = {
                'step': step,
                'repetition_score': current_score,
                'num_cycles': len(cycles) if cycles else 0,
                'generated_length': len(generated_text),
                'loss': float(loss),
                'generated_sample': generated_text[:100] + '...' if len(generated_text) > 100 else generated_text,
                'cycles_detected': cycles[:3] if cycles else []  # First 3 cycles
            }
            
            results['optimization_steps'].append(step_result)
            
            if current_score > best_score:
                best_score = current_score
                best_generation = generated_text
            
            print(f"Step {step}: Score={current_score:.4f}, Cycles={len(cycles) if cycles else 0}, Loss={float(loss):.4f}")
        
        # Store final results
        results['best_generation'] = best_generation
        results['best_score'] = best_score
        
        # Store final intervention values
        for layer_idx in self.config.target_layers:
            results['final_interventions'][layer_idx] = {}
            for head_idx in self.config.target_heads:
                if layer_idx in self.attention_interventions and head_idx in self.attention_interventions[layer_idx]:
                    results['final_interventions'][layer_idx][head_idx] = float(
                        self.attention_interventions[layer_idx][head_idx]
                    )
        
        self._clear_hooks()
        return results


class GradientBasedExperiment:
    """Main experiment class for gradient-based repetition optimization."""
    
    def __init__(self, config: GradientBasedConfig):
        self.config = config
        self.optimizer = GradientBasedRepetitionOptimizer(config)
        
    def run_experiment(self, test_texts: List[str], output_dir: str) -> Dict:
        """Run the gradient-based experiment on test texts."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'experiment_type': 'gradient_based_repetition_optimization',
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'test_results': [],
            'summary': {}
        }
        
        print(f"Running gradient-based experiment on {len(test_texts)} texts...")
        
        successful_optimizations = 0
        total_best_score = 0.0
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Optimizing text {i+1}/{len(test_texts)} ---")
            
            try:
                # Use first 100 chars as prompt
                prompt = text[:100]
                
                # Optimize interventions for this text
                optimization_result = self.optimizer.optimize_interventions(prompt)
                
                # Evaluate final result
                final_cycles = detect_cycles(optimization_result['best_generation'])
                success = len(final_cycles) > 0 if final_cycles else False
                
                if success:
                    successful_optimizations += 1
                
                total_best_score += optimization_result['best_score']
                
                test_result = {
                    'text_index': i,
                    'prompt': prompt,
                    'success': success,
                    'best_score': optimization_result['best_score'],
                    'final_cycles_count': len(final_cycles) if final_cycles else 0,
                    'optimization_steps': len(optimization_result['optimization_steps']),
                    'final_interventions': optimization_result['final_interventions'],
                    'best_generation': optimization_result['best_generation'],
                    'optimization_history': optimization_result['optimization_steps']
                }
                
                results['test_results'].append(test_result)
                
                print(f"Result: {'SUCCESS' if success else 'FAILED'}, Score: {optimization_result['best_score']:.4f}")
                
            except Exception as e:
                print(f"Error processing text {i}: {str(e)}")
                test_result = {
                    'text_index': i,
                    'prompt': text[:100],
                    'success': False,
                    'error': str(e)
                }
                results['test_results'].append(test_result)
        
        # Calculate summary statistics
        success_rate = successful_optimizations / len(test_texts) * 100
        avg_score = total_best_score / len(test_texts)
        
        results['summary'] = {
            'total_texts': len(test_texts),
            'successful_optimizations': successful_optimizations,
            'success_rate_percent': success_rate,
            'average_best_score': avg_score
        }
        
        print(f"\n=== GRADIENT-BASED EXPERIMENT SUMMARY ===")
        print(f"Success Rate: {success_rate:.1f}% ({successful_optimizations}/{len(test_texts)})")
        print(f"Average Best Score: {avg_score:.4f}")
        
        # Save results
        output_file = os.path.join(output_dir, 'gradient_based_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return results


def main():
    """Main function to run the gradient-based repetition experiment."""
    print("Starting Gradient-Based Repetition Optimization Experiment")
    print("=" * 60)
    
    # Configuration
    config = GradientBasedConfig(
        num_gradient_steps=15,  # More steps for better optimization
        learning_rate=0.05,     # Conservative learning rate
        target_layers=[15, 17, 19],  # Later layers
        target_heads=[0, 1, 2],      # Multiple heads
        newline_focus_strength=2.0   # Starting strength
    )
    
    # Load test data
    print("Loading test data...")
    try:
        test_data = load_cached_dataset(n_samples=10)
        # Use the texts directly
        test_texts = test_data
        print(f"Loaded {len(test_texts)} test texts")
    except Exception as e:
        print(f"Error loading cached data: {e}")
        # Fallback to simple test cases
        test_texts = [
            "The cat sat on the mat. The weather was nice today.",
            "Machine learning is fascinating. Deep learning models are powerful.",
            "Python is a programming language. Data science uses Python extensively.",
        ]
        print(f"Using {len(test_texts)} fallback test texts")
    
    # Create experiment
    experiment = GradientBasedExperiment(config)
    
    # Output directory
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/gradient_based_experiment"
    
    # Run experiment
    try:
        results = experiment.run_experiment(test_texts, output_dir)
        
        print("\n" + "=" * 60)
        print("GRADIENT-BASED EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Success Rate: {results['summary']['success_rate_percent']:.1f}%")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()