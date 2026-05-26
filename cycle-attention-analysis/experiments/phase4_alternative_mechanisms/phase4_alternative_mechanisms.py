#!/usr/bin/env python3
"""
Phase 4: Alternative Causal Mechanism Discovery

Since newlines are correlational not causal, we need to find the ACTUAL
mechanisms that trigger repetitive generation in transformer models.

Investigation areas:
1. Semantic content patterns that trigger repetition
2. Attention flow disruptions 
3. Context length dependencies
4. Specific token sequences that induce cycles
5. Model internal states during repetitive generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import sys
import re
from collections import Counter, defaultdict

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


def detect_repetition_standardized(text: str, tokenizer) -> Dict[str, Any]:
    """Standardized repetition analysis using token-based detect_cycles."""
    if not text or not text.strip():
        return {
            'total_cycles': 0, 
            'cycle_size': 0, 
            'cycle_count': 0,
            'cycle_pattern': None,
            'standardized': True
        }
    
    try:
        # Tokenize the text
        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        
        # Use standard detect_cycles function
        cycle, cycle_size, cycle_count = detect_cycles(tokens.tolist())
        
        # Decode cycle pattern if found
        cycle_pattern = None
        if cycle:
            cycle_pattern = tokenizer.decode(cycle, skip_special_tokens=True)
        
        return {
            'total_cycles': cycle_count if cycle_count else 0,
            'cycle_size': cycle_size,
            'cycle_count': cycle_count if cycle_count else 0, 
            'cycle_pattern': cycle_pattern,
            'standardized': True
        }
    except Exception as e:
        print(f"Warning: Error in standardized cycle detection: {e}")
        return {
            'total_cycles': 0,
            'cycle_size': 0,
            'cycle_count': 0, 
            'cycle_pattern': None,
            'standardized': True,
            'error': str(e)
        }


class AlternativeCausalMechanismExplorer:
    """Investigate alternative causal mechanisms for repetition induction."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_params = {
            'max_length': 300,
            'temperature': 0.8,
            'top_p': 0.9,
            'do_sample': True,
            'repetition_penalty': 1.0  # No penalty to allow natural repetition
        }
        
    def load_model(self):
        """Load model and setup."""
        print("Loading model...")
        try:
            self.model, self.tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def generate_with_analysis(self, prompt: str, return_tokens: bool = False) -> Dict[str, Any]:
        """Generate text and analyze for repetition patterns."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=min(len(input_ids[0]) + 200, self.generation_params['max_length']),
                temperature=self.generation_params['temperature'],
                top_p=self.generation_params['top_p'],
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=self.generation_params['do_sample'],
                repetition_penalty=self.generation_params['repetition_penalty']
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_only = generated_text[len(prompt):]
        
        # Analyze repetition
        repetition_analysis = detect_repetition_standardized(generated_only, self.tokenizer)
        
        result = {
            'prompt': prompt,
            'generated_text': generated_only,
            'repetition_analysis': repetition_analysis,
            'input_length': len(input_ids[0]),
            'output_length': len(output[0]) - len(input_ids[0])
        }
        
        if return_tokens:
            result['input_tokens'] = input_ids[0].tolist()
            result['output_tokens'] = output[0].tolist()
        
        return result
    
    def test_semantic_content_triggers(self) -> Dict[str, Any]:
        """Test if specific semantic content patterns trigger repetition."""
        print("  🧪 Testing Semantic Content Triggers")
        
        # Different semantic contexts that might trigger repetition
        test_prompts = [
            # Lists and enumerations
            "The main categories are: category A, category B, category C,",
            "Steps to follow: 1. First step, 2. Second step, 3. Third step,",
            "Items include: apple, banana, cherry, date,",
            
            # Repetitive instructions
            "To complete this task, you need to repeat the following process:",
            "The procedure involves doing the same thing over and over:",
            "This requires repeated application of the same method:",
            
            # Definitional contexts
            "The definition states that X is X when X exhibits",
            "According to the rule, A equals A if and only if A",
            "The theorem proves that Y is Y under conditions where Y",
            
            # Conversational patterns  
            "Person A said something, then Person B said something, then Person A said",
            "He told her that she told him that he told her",
            "The message was passed from X to Y to X to Y",
            
            # Technical repetitive contexts
            "The algorithm repeatedly applies the same transformation until",
            "Each iteration performs the identical operation on",
            "The loop continues executing the same code block"
        ]
        
        results = []
        
        for prompt in test_prompts:
            result = self.generate_with_analysis(prompt)
            result['category'] = self._categorize_prompt(prompt)
            results.append(result)
        
        return {
            'test_type': 'semantic_triggers',
            'results': results,
            'summary': self._analyze_semantic_results(results)
        }
    
    def test_context_length_dependencies(self) -> Dict[str, Any]:
        """Test if context length affects repetition probability."""
        print("  🧪 Testing Context Length Dependencies")
        
        base_text = "The quick brown fox jumps over the lazy dog. This is a sample sentence that will be repeated at different lengths to test context dependency effects."
        
        # Test different context lengths
        context_lengths = [10, 25, 50, 100, 150, 200]
        results = []
        
        for length in context_lengths:
            # Create context of specified length
            words = base_text.split()
            if length <= len(words):
                context = ' '.join(words[:length])
            else:
                # Repeat base text to reach desired length
                repeats_needed = (length // len(words)) + 1
                extended_text = (base_text + ' ') * repeats_needed
                context = ' '.join(extended_text.split()[:length])
            
            # Generate from this context
            result = self.generate_with_analysis(context)
            result['context_length'] = length
            result['context_words'] = len(context.split())
            results.append(result)
        
        return {
            'test_type': 'context_length',
            'results': results,
            'summary': self._analyze_length_dependency(results)
        }
    
    def test_token_sequence_patterns(self) -> Dict[str, Any]:
        """Test specific token sequences that might trigger repetition."""
        print("  🧪 Testing Token Sequence Patterns")
        
        # Test sequences that might trigger repetitive behavior
        pattern_tests = [
            # Punctuation patterns
            "... ... ... and then",
            "--- --- --- continuing with",
            "=== === === followed by",
            
            # Word patterns
            "again and again and again",
            "more and more and more",
            "over and over and over",
            
            # Number patterns  
            "1, 2, 3, 4, 5, 6, and then",
            "first, second, third, fourth, and then",
            "one by one by one by one",
            
            # Structural patterns
            "if A then B, if B then C, if C then",
            "not only X but also Y, not only Y but also",
            "either this or that, either that or",
            
            # Ending patterns that might loop
            "to be continued, to be continued, to be",
            "and so on, and so on, and so",
            "etc., etc., etc., and"
        ]
        
        results = []
        
        for pattern in pattern_tests:
            result = self.generate_with_analysis(pattern)
            result['pattern_type'] = self._categorize_pattern(pattern)
            results.append(result)
        
        return {
            'test_type': 'token_sequences', 
            'results': results,
            'summary': self._analyze_pattern_results(results)
        }
    
    def test_attention_disruption_hypothesis(self) -> Dict[str, Any]:
        """Test if attention disruption correlates with repetition."""
        print("  🧪 Testing Attention Disruption Hypothesis")
        
        # Create prompts designed to disrupt normal attention flow
        disruption_tests = [
            # Sudden topic changes
            "The weather was nice today. Quantum mechanics involves wave functions.",
            "She was cooking dinner when suddenly nuclear physics became relevant.",
            "The meeting discussed budget items. Dolphins communicate through echolocation.",
            
            # Contradictory statements
            "The sky is blue. The sky is not blue. The sky is green.",
            "All birds can fly. Penguins cannot fly. All penguins are birds.",
            "This statement is true. This statement is false. This statement is",
            
            # Incomplete or broken patterns
            "First, second, third, seventh, second, first,",
            "A leads to B leads to C leads to A leads to",
            "Input produces output produces input produces output produces",
            
            # Nested or recursive structures
            "The person who knows the person who knows the person who",
            "In the box that contains the box that contains the box",
            "According to the source that cites the source that cites"
        ]
        
        results = []
        
        for test_prompt in disruption_tests:
            result = self.generate_with_analysis(test_prompt)
            result['disruption_type'] = self._categorize_disruption(test_prompt)
            results.append(result)
        
        return {
            'test_type': 'attention_disruption',
            'results': results, 
            'summary': self._analyze_disruption_results(results)
        }
    
    def test_training_data_artifacts(self) -> Dict[str, Any]:
        """Test patterns that might come from training data artifacts."""
        print("  🧪 Testing Training Data Artifacts")
        
        # Common patterns from web text, code, and books that might be in training data
        artifact_tests = [
            # Web/HTML patterns
            "Click here for more information. Click here for",
            "Copyright 2023. All rights reserved. Copyright 2023.",
            "Loading... Please wait. Loading... Please wait.",
            
            # Code patterns
            "def function(): return function(). def function():",
            "for i in range(len(items)): for i in range(",
            "if condition: else: if condition: else:",
            
            # Book/document patterns
            "Chapter 1: Introduction. Chapter 2: Methods. Chapter 3:",
            "See page 123 for details. See page 124 for",
            "Figure 1 shows the results. Figure 2 shows",
            
            # Academic patterns
            "et al. (2023) found that Smith et al. (2024) found that",
            "The results indicate that the findings suggest that the results",
            "In conclusion, we conclude that in conclusion,",
            
            # FAQ/repetitive text patterns
            "Frequently asked questions: Q: What is A? A: A is B. Q: What is",
            "The answer is yes. The answer is no. The answer is",
            "Please note that users should be aware that please note"
        ]
        
        results = []
        
        for artifact in artifact_tests:
            result = self.generate_with_analysis(artifact)
            result['artifact_type'] = self._categorize_artifact(artifact)
            results.append(result)
        
        return {
            'test_type': 'training_artifacts',
            'results': results,
            'summary': self._analyze_artifact_results(results)
        }
    
    def _categorize_prompt(self, prompt: str) -> str:
        """Categorize semantic prompt types."""
        if 'category' in prompt.lower() or 'steps' in prompt.lower() or 'items' in prompt.lower():
            return 'enumeration'
        elif 'repeat' in prompt.lower() or 'same' in prompt.lower():
            return 'repetitive_instruction'
        elif 'definition' in prompt.lower() or 'equals' in prompt.lower() or 'theorem' in prompt.lower():
            return 'definitional'
        elif 'person' in prompt.lower() or 'said' in prompt.lower() or 'told' in prompt.lower():
            return 'conversational'
        elif 'algorithm' in prompt.lower() or 'iteration' in prompt.lower() or 'loop' in prompt.lower():
            return 'technical'
        else:
            return 'other'
    
    def _categorize_pattern(self, pattern: str) -> str:
        """Categorize token sequence pattern types."""
        if any(punct in pattern for punct in ['...', '---', '===']):
            return 'punctuation'
        elif 'again' in pattern or 'more' in pattern or 'over' in pattern:
            return 'word_repetition'
        elif any(num in pattern for num in ['1', '2', '3', 'first', 'second', 'third']):
            return 'numerical'
        elif 'if' in pattern or 'not only' in pattern or 'either' in pattern:
            return 'structural'
        elif 'continued' in pattern or 'so on' in pattern or 'etc' in pattern:
            return 'ending'
        else:
            return 'other'
    
    def _categorize_disruption(self, prompt: str) -> str:
        """Categorize attention disruption types."""
        if 'suddenly' in prompt.lower() or 'when' in prompt.lower():
            return 'topic_change'
        elif 'not' in prompt.lower() or 'false' in prompt.lower():
            return 'contradictory'
        elif 'first' in prompt.lower() and 'second' in prompt.lower():
            return 'broken_pattern'
        elif 'who' in prompt.lower() or 'that' in prompt.lower():
            return 'recursive'
        else:
            return 'other'
    
    def _categorize_artifact(self, artifact: str) -> str:
        """Categorize training data artifact types."""
        if 'click' in artifact.lower() or 'copyright' in artifact.lower() or 'loading' in artifact.lower():
            return 'web'
        elif 'def' in artifact.lower() or 'for' in artifact.lower() or 'if' in artifact.lower():
            return 'code'
        elif 'chapter' in artifact.lower() or 'page' in artifact.lower() or 'figure' in artifact.lower():
            return 'document'
        elif 'et al' in artifact.lower() or 'results' in artifact.lower() or 'conclusion' in artifact.lower():
            return 'academic'
        elif 'question' in artifact.lower() or 'answer' in artifact.lower() or 'please' in artifact.lower():
            return 'faq'
        else:
            return 'other'
    
    def _analyze_semantic_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze semantic trigger test results."""
        category_scores = defaultdict(list)
        
        for result in results:
            cycles = result['repetition_analysis']['total_cycles']
            category = result['category']
            category_scores[category].append(cycles)
        
        # Calculate averages per category
        category_averages = {}
        for category, scores in category_scores.items():
            category_averages[category] = {
                'avg_cycles': np.mean(scores),
                'max_cycles': max(scores),
                'count': len(scores)
            }
        
        # Find most repetitive category
        most_repetitive = max(category_averages.items(), 
                            key=lambda x: x[1]['avg_cycles'])
        
        return {
            'category_averages': category_averages,
            'most_repetitive_category': most_repetitive[0],
            'highest_avg_cycles': most_repetitive[1]['avg_cycles'],
            'total_tests': len(results)
        }
    
    def _analyze_length_dependency(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze context length dependency results."""
        length_cycles = [(r['context_length'], r['repetition_analysis']['total_cycles']) 
                        for r in results]
        
        # Calculate correlation between length and cycles
        lengths = [lc[0] for lc in length_cycles]
        cycles = [lc[1] for lc in length_cycles]
        
        correlation = np.corrcoef(lengths, cycles)[0, 1] if len(set(cycles)) > 1 else 0
        
        # Find optimal length range
        max_cycles_idx = cycles.index(max(cycles))
        optimal_length = lengths[max_cycles_idx]
        
        return {
            'length_cycle_pairs': length_cycles,
            'correlation': correlation,
            'optimal_length': optimal_length,
            'max_cycles': max(cycles),
            'length_dependency': abs(correlation) > 0.3
        }
    
    def _analyze_pattern_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze token sequence pattern results."""
        pattern_scores = defaultdict(list)
        
        for result in results:
            cycles = result['repetition_analysis']['total_cycles']
            pattern_type = result['pattern_type']
            pattern_scores[pattern_type].append(cycles)
        
        # Find most effective pattern type
        pattern_averages = {}
        for pattern_type, scores in pattern_scores.items():
            pattern_averages[pattern_type] = {
                'avg_cycles': np.mean(scores),
                'max_cycles': max(scores),
                'count': len(scores)
            }
        
        most_effective = max(pattern_averages.items(),
                           key=lambda x: x[1]['avg_cycles'])
        
        return {
            'pattern_averages': pattern_averages,
            'most_effective_pattern': most_effective[0],
            'highest_avg_cycles': most_effective[1]['avg_cycles'],
            'effective_patterns': [p for p, stats in pattern_averages.items() 
                                 if stats['avg_cycles'] > 0]
        }
    
    def _analyze_disruption_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze attention disruption results."""
        disruption_scores = defaultdict(list)
        
        for result in results:
            cycles = result['repetition_analysis']['total_cycles']
            disruption_type = result['disruption_type']
            disruption_scores[disruption_type].append(cycles)
        
        # Calculate averages per disruption type
        disruption_averages = {}
        for disruption_type, scores in disruption_scores.items():
            disruption_averages[disruption_type] = {
                'avg_cycles': np.mean(scores),
                'max_cycles': max(scores),
                'count': len(scores)
            }
        
        most_disruptive = max(disruption_averages.items(),
                            key=lambda x: x[1]['avg_cycles'])
        
        return {
            'disruption_averages': disruption_averages,
            'most_disruptive_type': most_disruptive[0], 
            'highest_avg_cycles': most_disruptive[1]['avg_cycles'],
            'disruption_effective': most_disruptive[1]['avg_cycles'] > 1
        }
    
    def _analyze_artifact_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze training data artifact results."""
        artifact_scores = defaultdict(list)
        
        for result in results:
            cycles = result['repetition_analysis']['total_cycles']
            artifact_type = result['artifact_type']
            artifact_scores[artifact_type].append(cycles)
        
        # Calculate averages per artifact type
        artifact_averages = {}
        for artifact_type, scores in artifact_scores.items():
            artifact_averages[artifact_type] = {
                'avg_cycles': np.mean(scores),
                'max_cycles': max(scores),
                'count': len(scores)
            }
        
        most_triggering = max(artifact_averages.items(),
                            key=lambda x: x[1]['avg_cycles'])
        
        return {
            'artifact_averages': artifact_averages,
            'most_triggering_artifact': most_triggering[0],
            'highest_avg_cycles': most_triggering[1]['avg_cycles'],
            'training_artifacts_effective': most_triggering[1]['avg_cycles'] > 2
        }
    
    def run_comprehensive_mechanism_discovery(self) -> Dict[str, Any]:
        """Run all alternative causal mechanism tests."""
        if not self.load_model():
            return {'error': 'Failed to load model'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'alternative_causal_mechanisms',
            'tests': {},
            'overall_summary': {}
        }
        
        print("🔬 PHASE 4: ALTERNATIVE CAUSAL MECHANISM DISCOVERY")
        print("=" * 70)
        print("Since newlines are correlational not causal, finding ACTUAL triggers...")
        print()
        
        # Run all test categories
        test_methods = [
            ('semantic_triggers', self.test_semantic_content_triggers),
            ('context_length', self.test_context_length_dependencies), 
            ('token_sequences', self.test_token_sequence_patterns),
            ('attention_disruption', self.test_attention_disruption_hypothesis),
            ('training_artifacts', self.test_training_data_artifacts)
        ]
        
        for test_name, test_method in test_methods:
            print(f"Running {test_name.replace('_', ' ').title()} tests...")
            try:
                test_result = test_method()
                results['tests'][test_name] = test_result
                
                # Print immediate summary
                summary = test_result['summary']
                if 'most_repetitive_category' in summary:
                    print(f"  → Most repetitive: {summary['most_repetitive_category']} "
                          f"({summary['highest_avg_cycles']:.1f} avg cycles)")
                elif 'most_effective_pattern' in summary:
                    print(f"  → Most effective: {summary['most_effective_pattern']} "
                          f"({summary['highest_avg_cycles']:.1f} avg cycles)")
                elif 'correlation' in summary:
                    print(f"  → Length correlation: {summary['correlation']:.3f}, "
                          f"optimal: {summary['optimal_length']} words")
                elif 'most_disruptive_type' in summary:
                    print(f"  → Most disruptive: {summary['most_disruptive_type']} "
                          f"({summary['highest_avg_cycles']:.1f} avg cycles)")
                elif 'most_triggering_artifact' in summary:
                    print(f"  → Most triggering: {summary['most_triggering_artifact']} "
                          f"({summary['highest_avg_cycles']:.1f} avg cycles)")
                
            except Exception as e:
                print(f"  ❌ Error in {test_name}: {e}")
                results['tests'][test_name] = {'error': str(e)}
        
        # Generate overall analysis
        results['overall_summary'] = self._generate_overall_analysis(results['tests'])
        
        return results
    
    def _generate_overall_analysis(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall analysis of all mechanism discovery tests."""
        
        # Collect all effectiveness scores
        effectiveness_scores = {}
        
        for test_name, test_data in test_results.items():
            if 'error' in test_data:
                continue
                
            summary = test_data.get('summary', {})
            
            if 'highest_avg_cycles' in summary:
                effectiveness_scores[test_name] = summary['highest_avg_cycles']
        
        # Find most effective mechanism category
        if effectiveness_scores:
            most_effective_test = max(effectiveness_scores.items(), key=lambda x: x[1])
            
            # Determine if any mechanism shows strong evidence
            max_effectiveness = most_effective_test[1]
            
            if max_effectiveness >= 5:
                evidence_strength = "STRONG"
                conclusion = f"🎯 BREAKTHROUGH: {most_effective_test[0].replace('_', ' ').title()} shows strong causal evidence!"
                recommendation = f"Focus intensive investigation on {most_effective_test[0]} mechanisms"
            elif max_effectiveness >= 2:
                evidence_strength = "MODERATE" 
                conclusion = f"📈 PROMISING: {most_effective_test[0].replace('_', ' ').title()} shows moderate evidence"
                recommendation = f"Investigate {most_effective_test[0]} mechanisms further with targeted experiments"
            else:
                evidence_strength = "WEAK"
                conclusion = "❌ Limited evidence found across all tested mechanisms"
                recommendation = "Consider alternative approaches or deeper model analysis"
        else:
            evidence_strength = "NONE"
            conclusion = "❌ No clear causal mechanisms identified"
            recommendation = "Fundamental approach change needed - consider model internals analysis"
        
        return {
            'effectiveness_scores': effectiveness_scores,
            'most_effective_mechanism': most_effective_test[0] if effectiveness_scores else None,
            'max_effectiveness_score': most_effective_test[1] if effectiveness_scores else 0,
            'evidence_strength': evidence_strength,
            'conclusion': conclusion,
            'recommendation': recommendation,
            'next_phase_suggested': evidence_strength in ['STRONG', 'MODERATE']
        }


def main():
    """Main experiment function."""
    print("🚀 PHASE 4: ALTERNATIVE CAUSAL MECHANISM DISCOVERY")
    print("=" * 70)
    
    # Run comprehensive mechanism discovery
    explorer = AlternativeCausalMechanismExplorer()
    results = explorer.run_comprehensive_mechanism_discovery()
    
    if 'error' in results:
        print(f"❌ Experiments failed: {results['error']}")
        return False
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE MECHANISM DISCOVERY SUMMARY")
    print("=" * 70)
    
    overall = results['overall_summary']
    print(f"\n{overall['conclusion']}")
    print(f"\nEvidence Strength: {overall['evidence_strength']}")
    print(f"Recommendation: {overall['recommendation']}")
    
    if overall['effectiveness_scores']:
        print(f"\nMechanism Effectiveness Ranking:")
        sorted_scores = sorted(overall['effectiveness_scores'].items(), 
                              key=lambda x: x[1], reverse=True)
        for i, (mechanism, score) in enumerate(sorted_scores, 1):
            print(f"  {i}. {mechanism.replace('_', ' ').title()}: {score:.1f} avg cycles")
    
    # Save results
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/mechanism_discovery"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'alternative_mechanisms_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    if overall['next_phase_suggested']:
        print(f"\n🎉 BREAKTHROUGH DETECTED!")
        print(f"🎯 Proceed to targeted investigation of: {overall['most_effective_mechanism']}")
    else:
        print(f"\n🔍 Continue systematic investigation...")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)