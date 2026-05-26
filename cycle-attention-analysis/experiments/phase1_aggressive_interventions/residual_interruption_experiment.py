#!/usr/bin/env python3
"""
Residual Stream Interruption Experiment

This experiment hooks residual connections to force repetitive patterns
in hidden states by interrupting the residual stream and copying patterns.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
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
class ResidualInterruptionConfig:
    """Configuration for residual stream interruption."""
    model_name: str = "EleutherAI/pythia-1.4b"
    max_length: int = 512
    num_generate: int = 100
    
    # Residual interruption parameters
    target_layers: List[int] = None  # If None, use [15, 17, 19, 21]
    interruption_strength: float = 1.0
    copy_window_size: int = 6
    copy_distance: int = 10
    
    # Interruption strategies
    strategies: List[str] = None  # ["direct_copy", "pattern_repeat", "activation_echo", "gradient_copy"]
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [15, 17, 19, 21]
        if self.strategies is None:
            self.strategies = ["direct_copy", "pattern_repeat", "activation_echo", "gradient_copy"]


class ResidualStreamInterruptor:
    """Interrupts residual streams to force repetitive patterns."""
    
    def __init__(self, config: ResidualInterruptionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.activation_cache = {}
        
        # Initialize model
        self._initialize_model()
        
        # Get model dimensions
        self.hidden_size = self.model.config.hidden_size
        print(f"Hidden size: {self.hidden_size}")
        
    def _initialize_model(self):
        """Load model and tokenizer."""
        print("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(self.config.model_name)
        self.model.eval()
        print(f"Model loaded: {self.config.model_name}")
    
    def _clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_cache = {}
    
    def create_direct_copy_hook(self, layer_idx: int) -> Callable:
        """Create hook that directly copies earlier activations to later positions."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Copy from earlier positions to later positions
            copy_distance = min(self.config.copy_distance, seq_len // 2)
            copy_window = min(self.config.copy_window_size, copy_distance)
            
            if seq_len > copy_distance + copy_window:
                for i in range(copy_window):
                    source_pos = i
                    target_pos = copy_distance + i
                    
                    if target_pos < seq_len:
                        # Direct copy with strength modulation
                        strength = self.config.interruption_strength
                        hidden_states[:, target_pos, :] = (
                            (1 - strength) * hidden_states[:, target_pos, :] +
                            strength * hidden_states[:, source_pos, :].clone()
                        )
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return hook_fn
    
    def create_pattern_repeat_hook(self, layer_idx: int) -> Callable:
        """Create hook that repeats patterns at regular intervals."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            pattern_length = self.config.copy_window_size
            repeat_interval = pattern_length * 2
            
            # Extract a pattern from early in the sequence
            if seq_len > pattern_length * 3:
                pattern_start = seq_len // 4
                pattern_end = pattern_start + pattern_length
                
                if pattern_end < seq_len:
                    pattern = hidden_states[:, pattern_start:pattern_end, :].clone()
                    
                    # Repeat this pattern at intervals
                    repeat_start = seq_len // 2
                    pos = repeat_start
                    
                    while pos + pattern_length <= seq_len:
                        strength = self.config.interruption_strength * 0.8  # Slightly weaker
                        
                        for i in range(pattern_length):
                            if pos + i < seq_len:
                                hidden_states[:, pos + i, :] = (
                                    (1 - strength) * hidden_states[:, pos + i, :] +
                                    strength * pattern[:, i, :]
                                )
                        
                        pos += repeat_interval
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return hook_fn
    
    def create_activation_echo_hook(self, layer_idx: int) -> Callable:
        """Create hook that creates echoes of activations with decay."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Create echo effect - each position gets influenced by previous positions
            echo_distance = min(self.config.copy_distance // 2, seq_len // 3)
            
            if echo_distance > 0:
                for target_pos in range(echo_distance, seq_len):
                    echo_sources = []
                    echo_weights = []
                    
                    # Collect multiple echo sources with decay
                    for i in range(1, min(4, echo_distance + 1)):
                        source_pos = target_pos - i * (echo_distance // 3)
                        if source_pos >= 0:
                            echo_sources.append(hidden_states[:, source_pos, :].clone())
                            # Exponential decay for echo strength
                            weight = self.config.interruption_strength * (0.7 ** i)
                            echo_weights.append(weight)
                    
                    if echo_sources:
                        # Combine echoes
                        total_echo_weight = sum(echo_weights)
                        if total_echo_weight > 0:
                            echo_contribution = torch.zeros_like(hidden_states[:, target_pos, :])
                            
                            for source, weight in zip(echo_sources, echo_weights):
                                echo_contribution += (weight / total_echo_weight) * source
                            
                            # Blend with original
                            blend_strength = min(total_echo_weight, 0.6)
                            hidden_states[:, target_pos, :] = (
                                (1 - blend_strength) * hidden_states[:, target_pos, :] +
                                blend_strength * echo_contribution
                            )
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return hook_fn
    
    def create_gradient_copy_hook(self, layer_idx: int) -> Callable:
        """Create hook that uses gradient-like copying for smooth transitions."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Create smooth repetitive gradients
            if seq_len > self.config.copy_window_size * 2:
                # Define regions for gradient copying
                source_start = seq_len // 5
                source_end = source_start + self.config.copy_window_size
                
                target_start = seq_len * 3 // 5
                
                if source_end < seq_len and target_start + self.config.copy_window_size <= seq_len:
                    # Extract source pattern
                    source_pattern = hidden_states[:, source_start:source_end, :].clone()
                    
                    # Apply with gradient-based blending
                    for i in range(self.config.copy_window_size):
                        target_pos = target_start + i
                        
                        if target_pos < seq_len:
                            # Gradient strength increases towards middle of pattern
                            progress = i / (self.config.copy_window_size - 1)
                            gradient_strength = self.config.interruption_strength * (
                                1.0 - abs(2 * progress - 1)  # Peak at middle
                            )
                            
                            hidden_states[:, target_pos, :] = (
                                (1 - gradient_strength) * hidden_states[:, target_pos, :] +
                                gradient_strength * source_pattern[:, i, :]
                            )
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return hook_fn
    
    def install_interruption_hooks(self, strategy: str):
        """Install residual stream interruption hooks for given strategy."""
        self._clear_hooks()
        
        hook_creators = {
            "direct_copy": self.create_direct_copy_hook,
            "pattern_repeat": self.create_pattern_repeat_hook,
            "activation_echo": self.create_activation_echo_hook,
            "gradient_copy": self.create_gradient_copy_hook
        }
        
        if strategy not in hook_creators:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        hook_creator = hook_creators[strategy]
        
        for layer_idx in self.config.target_layers:
            if layer_idx < len(self.model.gpt_neox.layers):
                layer = self.model.gpt_neox.layers[layer_idx]
                hook = layer.register_forward_hook(hook_creator(layer_idx))
                self.hooks.append(hook)
        
        print(f"Installed {strategy} hooks on layers: {self.config.target_layers}")
    
    def generate_with_interruption(self, prompt: str, strategy: str) -> str:
        """Generate text with residual stream interruption."""
        # Install hooks for this strategy
        self.install_interruption_hooks(strategy)
        
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            
            # Generate with interruption
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
            
        finally:
            # Always clear hooks
            self._clear_hooks()


class ResidualInterruptionExperiment:
    """Main experiment class for residual stream interruption."""
    
    def __init__(self, config: ResidualInterruptionConfig):
        self.config = config
        self.interruptor = ResidualStreamInterruptor(config)
    
    def test_single_strategy(self, prompt: str, strategy: str) -> Dict:
        """Test a single interruption strategy on a given prompt."""
        print(f"Testing strategy '{strategy}' on prompt: {prompt[:50]}...")
        
        try:
            # Generate with interruption
            generated_text = self.interruptor.generate_with_interruption(prompt, strategy)
            
            # Analyze for cycles
            cycles = detect_cycles(generated_text)
            success = len(cycles) > 0 if cycles else False
            
            # Calculate repetition score
            if cycles:
                total_score = sum(len(cycle['repeated_text']) * cycle['repetitions'] 
                                for cycle in cycles)
                repetition_score = float(total_score) / len(generated_text)
            else:
                repetition_score = 0.0
            
            return {
                'strategy': strategy,
                'success': success,
                'repetition_score': repetition_score,
                'num_cycles': len(cycles) if cycles else 0,
                'generated_text': generated_text,
                'cycles_detected': cycles[:5] if cycles else [],  # First 5 cycles
                'generated_length': len(generated_text)
            }
            
        except Exception as e:
            print(f"Error testing strategy {strategy}: {str(e)}")
            return {
                'strategy': strategy,
                'success': False,
                'error': str(e)
            }
    
    def run_experiment(self, test_texts: List[str], output_dir: str) -> Dict:
        """Run the residual interruption experiment."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'experiment_type': 'residual_stream_interruption',
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'test_results': [],
            'strategy_summaries': {},
            'summary': {}
        }
        
        print(f"Running residual interruption experiment on {len(test_texts)} texts...")
        print(f"Testing strategies: {self.config.strategies}")
        
        # Initialize strategy summaries
        for strategy in self.config.strategies:
            results['strategy_summaries'][strategy] = {
                'successes': 0,
                'total_tests': 0,
                'total_score': 0.0,
                'success_rate': 0.0
            }
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Testing text {i+1}/{len(test_texts)} ---")
            prompt = text[:100]  # Use first 100 chars as prompt
            
            text_results = {
                'text_index': i,
                'prompt': prompt,
                'strategy_results': {}
            }
            
            # Test each strategy
            for strategy in self.config.strategies:
                strategy_result = self.test_single_strategy(prompt, strategy)
                text_results['strategy_results'][strategy] = strategy_result
                
                # Update strategy summary
                summary = results['strategy_summaries'][strategy]
                summary['total_tests'] += 1
                summary['total_score'] += strategy_result.get('repetition_score', 0.0)
                
                if strategy_result.get('success', False):
                    summary['successes'] += 1
                
                print(f"  {strategy}: {'SUCCESS' if strategy_result.get('success') else 'FAILED'} "
                      f"(score: {strategy_result.get('repetition_score', 0.0):.4f})")
            
            results['test_results'].append(text_results)
        
        # Calculate final summaries
        total_successes = 0
        total_tests = 0
        
        for strategy, summary in results['strategy_summaries'].items():
            if summary['total_tests'] > 0:
                summary['success_rate'] = summary['successes'] / summary['total_tests'] * 100
                summary['average_score'] = summary['total_score'] / summary['total_tests']
            
            total_successes += summary['successes']
            total_tests += summary['total_tests']
        
        overall_success_rate = total_successes / total_tests * 100 if total_tests > 0 else 0.0
        
        results['summary'] = {
            'total_tests': total_tests,
            'total_successes': total_successes,
            'overall_success_rate': overall_success_rate,
            'best_strategy': max(results['strategy_summaries'].items(), 
                               key=lambda x: x[1]['success_rate'])[0] if results['strategy_summaries'] else None
        }
        
        print(f"\n=== RESIDUAL INTERRUPTION EXPERIMENT SUMMARY ===")
        print(f"Overall Success Rate: {overall_success_rate:.1f}% ({total_successes}/{total_tests})")
        
        for strategy, summary in results['strategy_summaries'].items():
            print(f"{strategy}: {summary['success_rate']:.1f}% "
                  f"({summary['successes']}/{summary['total_tests']}) "
                  f"avg_score: {summary.get('average_score', 0.0):.4f}")
        
        # Save results
        output_file = os.path.join(output_dir, 'residual_interruption_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return results


def main():
    """Main function to run the residual interruption experiment."""
    print("Starting Residual Stream Interruption Experiment")
    print("=" * 60)
    
    # Configuration
    config = ResidualInterruptionConfig(
        target_layers=[15, 17, 19, 21],  # Later layers for stronger effect
        interruption_strength=0.8,      # Strong interruption
        copy_window_size=5,             # Moderate window
        copy_distance=8,                # Reasonable distance
        strategies=["direct_copy", "pattern_repeat", "activation_echo", "gradient_copy"],
        temperature=0.8  # Slightly higher for variation
    )
    
    # Load test data
    print("Loading test data...")
    try:
        test_data = load_cached_dataset(n_samples=6)
        # Use the texts directly (4 strategies × 6 texts = 24 total tests)
        test_texts = test_data
        print(f"Loaded {len(test_texts)} test texts")
    except Exception as e:
        print(f"Error loading cached data: {e}")
        # Fallback to simple test cases
        test_texts = [
            "The cat sat on the mat. The weather was nice today.",
            "Machine learning is fascinating. Deep learning models are powerful.",
            "Python is a programming language. Data science uses Python extensively.",
            "The quick brown fox jumps over the lazy dog in the garden."
        ]
        print(f"Using {len(test_texts)} fallback test texts")
    
    # Create experiment
    experiment = ResidualInterruptionExperiment(config)
    
    # Output directory
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/residual_interruption_experiment"
    
    # Run experiment
    try:
        results = experiment.run_experiment(test_texts, output_dir)
        
        print("\n" + "=" * 60)
        print("RESIDUAL INTERRUPTION EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.1f}%")
        if results['summary']['best_strategy']:
            print(f"Best Strategy: {results['summary']['best_strategy']}")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()