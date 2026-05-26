#!/usr/bin/env python3
"""
Direct Embedding Manipulation Experiment

This experiment bypasses attention mechanisms entirely by directly modifying
token embeddings to inject repetitive patterns at the embedding level.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
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
class EmbeddingManipulationConfig:
    """Configuration for direct embedding manipulation."""
    model_name: str = "EleutherAI/pythia-1.4b"
    max_length: int = 512
    num_generate: int = 100
    
    # Embedding manipulation parameters
    repetition_strength: float = 0.5
    injection_positions: List[str] = None  # ["end", "middle", "periodic"]
    repetition_window_size: int = 5
    embedding_blend_ratio: float = 0.3
    
    # Pattern types to inject
    pattern_types: List[str] = None  # ["token_repeat", "phrase_repeat", "positional_copy"]
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    def __post_init__(self):
        if self.injection_positions is None:
            self.injection_positions = ["end", "middle", "periodic"]
        if self.pattern_types is None:
            self.pattern_types = ["token_repeat", "phrase_repeat", "positional_copy"]


class EmbeddingManipulator:
    """Manipulates embeddings directly to encourage repetitive patterns."""
    
    def __init__(self, config: EmbeddingManipulationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.original_embedding_forward = None
        
        # Initialize model
        self._initialize_model()
        
        # Get embedding dimension
        self.embedding_dim = self.model.gpt_neox.embed_in.embedding_dim
        print(f"Embedding dimension: {self.embedding_dim}")
        
    def _initialize_model(self):
        """Load model and tokenizer."""
        print("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(self.config.model_name)
        self.model.eval()
        print(f"Model loaded: {self.config.model_name}")
        
        # Store original embedding forward method
        self.original_embedding_forward = self.model.gpt_neox.embed_in.forward
    
    def create_repetitive_embeddings(self, input_ids: torch.Tensor, pattern_type: str) -> torch.Tensor:
        """Create embeddings with injected repetitive patterns."""
        batch_size, seq_len = input_ids.shape
        
        # Get original embeddings
        original_embeddings = self.original_embedding_forward(input_ids)
        
        if pattern_type == "token_repeat":
            return self._create_token_repeat_pattern(original_embeddings, input_ids)
        elif pattern_type == "phrase_repeat":
            return self._create_phrase_repeat_pattern(original_embeddings, input_ids)
        elif pattern_type == "positional_copy":
            return self._create_positional_copy_pattern(original_embeddings, input_ids)
        else:
            return original_embeddings
    
    def _create_token_repeat_pattern(self, embeddings: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Inject single token repetition patterns."""
        batch_size, seq_len, embed_dim = embeddings.shape
        modified_embeddings = embeddings.clone()
        
        # Strategy: Make later tokens similar to earlier high-frequency tokens
        for batch_idx in range(batch_size):
            # Find most frequent tokens in sequence
            unique_tokens, counts = torch.unique(input_ids[batch_idx], return_counts=True)
            if len(unique_tokens) > 1:
                # Get most frequent token (excluding potential padding)
                most_frequent_idx = torch.argmax(counts)
                most_frequent_token = unique_tokens[most_frequent_idx]
                
                # Find positions of this token
                token_positions = (input_ids[batch_idx] == most_frequent_token).nonzero(as_tuple=True)[0]
                
                if len(token_positions) > 0:
                    # Get the embedding of this frequent token
                    frequent_embedding = embeddings[batch_idx, token_positions[0]]
                    
                    # Inject this embedding into later positions
                    injection_positions = self._get_injection_positions(seq_len, "token_repeat")
                    for pos in injection_positions:
                        if pos < seq_len:
                            # Blend with original embedding
                            modified_embeddings[batch_idx, pos] = (
                                (1 - self.config.embedding_blend_ratio) * modified_embeddings[batch_idx, pos] +
                                self.config.embedding_blend_ratio * frequent_embedding
                            )
        
        return modified_embeddings
    
    def _create_phrase_repeat_pattern(self, embeddings: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Inject phrase-level repetition patterns."""
        batch_size, seq_len, embed_dim = embeddings.shape
        modified_embeddings = embeddings.clone()
        
        window_size = min(self.config.repetition_window_size, seq_len // 3)
        
        for batch_idx in range(batch_size):
            # Take a window from early in the sequence
            if seq_len > window_size * 2:
                source_start = seq_len // 4  # Start from 1/4 through
                source_end = source_start + window_size
                
                if source_end < seq_len:
                    source_embeddings = embeddings[batch_idx, source_start:source_end]
                    
                    # Inject this pattern at multiple later positions
                    injection_starts = self._get_phrase_injection_positions(seq_len, window_size)
                    
                    for inject_start in injection_starts:
                        inject_end = inject_start + window_size
                        if inject_end <= seq_len:
                            # Blend source pattern with target positions
                            for i, source_embed in enumerate(source_embeddings):
                                target_pos = inject_start + i
                                modified_embeddings[batch_idx, target_pos] = (
                                    (1 - self.config.embedding_blend_ratio) * modified_embeddings[batch_idx, target_pos] +
                                    self.config.embedding_blend_ratio * source_embed
                                )
        
        return modified_embeddings
    
    def _create_positional_copy_pattern(self, embeddings: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Create exact positional copies to force repetition."""
        batch_size, seq_len, embed_dim = embeddings.shape
        modified_embeddings = embeddings.clone()
        
        copy_distance = min(self.config.repetition_window_size * 2, seq_len // 3)
        
        for batch_idx in range(batch_size):
            # Copy embeddings from earlier positions to later positions
            for target_pos in range(copy_distance, seq_len):
                source_pos = target_pos - copy_distance
                
                # Strong blending for exact copying
                strong_blend_ratio = min(self.config.embedding_blend_ratio * 2, 0.8)
                modified_embeddings[batch_idx, target_pos] = (
                    (1 - strong_blend_ratio) * modified_embeddings[batch_idx, target_pos] +
                    strong_blend_ratio * embeddings[batch_idx, source_pos]
                )
        
        return modified_embeddings
    
    def _get_injection_positions(self, seq_len: int, pattern_type: str) -> List[int]:
        """Get positions where to inject repetitive patterns."""
        positions = []
        
        for position_type in self.config.injection_positions:
            if position_type == "end":
                # Inject in last 1/3 of sequence
                start_pos = seq_len * 2 // 3
                positions.extend(range(start_pos, seq_len, 2))
            elif position_type == "middle":
                # Inject in middle 1/3 of sequence
                start_pos = seq_len // 3
                end_pos = seq_len * 2 // 3
                positions.extend(range(start_pos, end_pos, 3))
            elif position_type == "periodic":
                # Inject at regular intervals
                interval = max(1, seq_len // 10)
                positions.extend(range(interval, seq_len, interval))
        
        # Remove duplicates and sort
        return sorted(list(set(positions)))
    
    def _get_phrase_injection_positions(self, seq_len: int, window_size: int) -> List[int]:
        """Get starting positions for phrase injections."""
        positions = []
        
        # Inject phrases in latter half of sequence
        start_search = seq_len // 2
        
        # Space injections to avoid overlap
        spacing = window_size + 2
        pos = start_search
        
        while pos + window_size <= seq_len:
            positions.append(pos)
            pos += spacing
        
        return positions
    
    def install_embedding_hook(self, pattern_type: str):
        """Install hook to manipulate embeddings during generation."""
        def modified_embedding_forward(input_ids):
            return self.create_repetitive_embeddings(input_ids, pattern_type)
        
        # Replace the forward method
        self.model.gpt_neox.embed_in.forward = modified_embedding_forward
    
    def restore_original_embedding(self):
        """Restore original embedding behavior."""
        self.model.gpt_neox.embed_in.forward = self.original_embedding_forward
    
    def generate_with_embedding_manipulation(self, prompt: str, pattern_type: str) -> str:
        """Generate text with embedding manipulation."""
        # Install embedding hook
        self.install_embedding_hook(pattern_type)
        
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            
            # Generate with manipulated embeddings
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
            # Always restore original behavior
            self.restore_original_embedding()


class EmbeddingManipulationExperiment:
    """Main experiment class for embedding manipulation."""
    
    def __init__(self, config: EmbeddingManipulationConfig):
        self.config = config
        self.manipulator = EmbeddingManipulator(config)
    
    def test_single_pattern(self, prompt: str, pattern_type: str) -> Dict:
        """Test a single pattern type on a given prompt."""
        print(f"Testing pattern '{pattern_type}' on prompt: {prompt[:50]}...")
        
        try:
            # Generate with manipulation
            generated_text = self.manipulator.generate_with_embedding_manipulation(
                prompt, pattern_type
            )
            
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
                'pattern_type': pattern_type,
                'success': success,
                'repetition_score': repetition_score,
                'num_cycles': len(cycles) if cycles else 0,
                'generated_text': generated_text,
                'cycles_detected': cycles[:5] if cycles else [],  # First 5 cycles
                'generated_length': len(generated_text)
            }
            
        except Exception as e:
            print(f"Error testing pattern {pattern_type}: {str(e)}")
            return {
                'pattern_type': pattern_type,
                'success': False,
                'error': str(e)
            }
    
    def run_experiment(self, test_texts: List[str], output_dir: str) -> Dict:
        """Run the embedding manipulation experiment."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'experiment_type': 'direct_embedding_manipulation',
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'test_results': [],
            'pattern_summaries': {},
            'summary': {}
        }
        
        print(f"Running embedding manipulation experiment on {len(test_texts)} texts...")
        print(f"Testing patterns: {self.config.pattern_types}")
        
        # Initialize pattern summaries
        for pattern in self.config.pattern_types:
            results['pattern_summaries'][pattern] = {
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
                'pattern_results': {}
            }
            
            # Test each pattern type
            for pattern_type in self.config.pattern_types:
                pattern_result = self.test_single_pattern(prompt, pattern_type)
                text_results['pattern_results'][pattern_type] = pattern_result
                
                # Update pattern summary
                summary = results['pattern_summaries'][pattern_type]
                summary['total_tests'] += 1
                summary['total_score'] += pattern_result.get('repetition_score', 0.0)
                
                if pattern_result.get('success', False):
                    summary['successes'] += 1
                
                print(f"  {pattern_type}: {'SUCCESS' if pattern_result.get('success') else 'FAILED'} "
                      f"(score: {pattern_result.get('repetition_score', 0.0):.4f})")
            
            results['test_results'].append(text_results)
        
        # Calculate final summaries
        total_successes = 0
        total_tests = 0
        
        for pattern, summary in results['pattern_summaries'].items():
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
            'best_pattern': max(results['pattern_summaries'].items(), 
                              key=lambda x: x[1]['success_rate'])[0] if results['pattern_summaries'] else None
        }
        
        print(f"\n=== EMBEDDING MANIPULATION EXPERIMENT SUMMARY ===")
        print(f"Overall Success Rate: {overall_success_rate:.1f}% ({total_successes}/{total_tests})")
        
        for pattern, summary in results['pattern_summaries'].items():
            print(f"{pattern}: {summary['success_rate']:.1f}% "
                  f"({summary['successes']}/{summary['total_tests']}) "
                  f"avg_score: {summary.get('average_score', 0.0):.4f}")
        
        # Save results
        output_file = os.path.join(output_dir, 'embedding_manipulation_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return results


def main():
    """Main function to run the embedding manipulation experiment."""
    print("Starting Direct Embedding Manipulation Experiment")
    print("=" * 60)
    
    # Configuration
    config = EmbeddingManipulationConfig(
        repetition_strength=0.7,
        injection_positions=["end", "middle", "periodic"],
        repetition_window_size=4,
        embedding_blend_ratio=0.4,
        pattern_types=["token_repeat", "phrase_repeat", "positional_copy"],
        temperature=0.8  # Slightly higher for more variation
    )
    
    # Load test data
    print("Loading test data...")
    try:
        test_data = load_cached_dataset(n_samples=8)
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
            "The quick brown fox jumps over the lazy dog repeatedly."
        ]
        print(f"Using {len(test_texts)} fallback test texts")
    
    # Create experiment
    experiment = EmbeddingManipulationExperiment(config)
    
    # Output directory
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/embedding_manipulation_experiment"
    
    # Run experiment
    try:
        results = experiment.run_experiment(test_texts, output_dir)
        
        print("\n" + "=" * 60)
        print("EMBEDDING MANIPULATION EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.1f}%")
        if results['summary']['best_pattern']:
            print(f"Best Pattern: {results['summary']['best_pattern']}")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()