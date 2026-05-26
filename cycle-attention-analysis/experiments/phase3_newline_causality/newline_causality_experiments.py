#!/usr/bin/env python3
"""
Phase 3: Direct Newline Token Interventions

Testing if newline token (ID 187) causally drives repetition through:
1. Token replacement interventions
2. Embedding manipulation
3. Attention blocking
4. Newline amplification
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add paths for imports
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu/modules')
sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src/modules')

try:
    from model_utils import load_model_and_tokenizer
    from cached_data_utils import load_cached_dataset
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Import standard cycle detection
try:
    from parrots.cycle_detection import detect_cycles
except ImportError:
    print("❌ Could not import detect_cycles from parrots.cycle_detection")
    sys.exit(1)


def count_cycles_standardized(text: str, tokenizer) -> int:
    """Standardized cycle detection using token-based approach."""
    if not text or not text.strip():
        return 0
    
    try:
        # Tokenize the text
        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        
        # Use standard detect_cycles function
        cycle, cycle_size, cycle_count = detect_cycles(tokens.tolist())
        
        return cycle_count if cycle_count else 0
    except Exception as e:
        print(f"Warning: Error in cycle detection: {e}")
        return 0


class NewlineInterventionExperiment:
    """Direct newline token intervention experiments."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.newline_token_id = None
        self.original_embedding = None
        
    def load_model(self):
        """Load model and setup."""
        print("Loading model...")
        try:
            self.model, self.tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get newline token
            self.newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
            
            # Store original embedding
            self.original_embedding = self.model.gpt_neox.embed_in.weight[self.newline_token_id].clone()
            
            print(f"✅ Model loaded. Newline token ID: {self.newline_token_id}")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def generate_text(self, prompt: str, max_length: int = 300) -> str:
        """Generate text from prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                repetition_penalty=1.0  # No repetition penalty to allow natural repetition
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
    
    def test_newline_removal(self, text: str) -> Dict[str, Any]:
        """Test 1: Remove newlines and compare repetition."""
        print("  🧪 Test 1: Newline Removal")
        
        # Original with newlines
        original_prompt = text[:100]
        original_gen = self.generate_text(original_prompt)
        original_cycles = count_cycles_standardized(original_gen, self.tokenizer)
        
        # Without newlines (replace with spaces)
        no_newline_prompt = original_prompt.replace('\n', ' ')
        no_newline_gen = self.generate_text(no_newline_prompt)
        no_newline_cycles = count_cycles_standardized(no_newline_gen, self.tokenizer)
        
        return {
            'original_cycles': original_cycles,
            'no_newline_cycles': no_newline_cycles,
            'reduction': original_cycles - no_newline_cycles,
            'causal_evidence': no_newline_cycles < original_cycles,
            'original_sample': original_gen[:150],
            'no_newline_sample': no_newline_gen[:150]
        }
    
    def test_newline_amplification(self, text: str) -> Dict[str, Any]:
        """Test 2: Add more newlines and compare repetition."""
        print("  🧪 Test 2: Newline Amplification")
        
        # Original
        original_prompt = text[:100]
        original_gen = self.generate_text(original_prompt)
        original_cycles = count_cycles_standardized(original_gen, self.tokenizer)
        
        # Double newlines
        double_prompt = original_prompt.replace('\n', '\n\n')
        double_gen = self.generate_text(double_prompt)
        double_cycles = count_cycles_standardized(double_gen, self.tokenizer)
        
        # Triple newlines  
        triple_prompt = original_prompt.replace('\n', '\n\n\n')
        triple_gen = self.generate_text(triple_prompt)
        triple_cycles = count_cycles_standardized(triple_gen, self.tokenizer)
        
        return {
            'original_cycles': original_cycles,
            'double_cycles': double_cycles,
            'triple_cycles': triple_cycles,
            'amplification_effect': triple_cycles > original_cycles,
            'dose_response': [original_cycles, double_cycles, triple_cycles],
            'triple_sample': triple_gen[:150]
        }
    
    def test_token_replacement(self, text: str) -> Dict[str, Any]:
        """Test 3: Replace newline tokens with other tokens."""
        print("  🧪 Test 3: Token Replacement")
        
        # Get replacement token IDs
        replacements = {
            'space': self.tokenizer.encode(' ', add_special_tokens=False)[0],
            'period': self.tokenizer.encode('.', add_special_tokens=False)[0],
            'semicolon': self.tokenizer.encode(';', add_special_tokens=False)[0],
            'tab': self.tokenizer.encode('\t', add_special_tokens=False)[0] if '\t' in self.tokenizer.get_vocab() else None
        }
        
        # Original
        original_prompt = text[:100]
        original_gen = self.generate_text(original_prompt)
        original_cycles = count_cycles_standardized(original_gen, self.tokenizer)
        
        results = {'original_cycles': original_cycles}
        
        for name, replacement_id in replacements.items():
            if replacement_id is None:
                continue
                
            # Replace in prompt
            modified_prompt = original_prompt.replace('\n', self.tokenizer.decode([replacement_id]))
            modified_gen = self.generate_text(modified_prompt)
            modified_cycles = count_cycles_standardized(modified_gen, self.tokenizer)
            
            results[f'{name}_cycles'] = modified_cycles
            results[f'{name}_sample'] = modified_gen[:150]
        
        return results
    
    def test_embedding_manipulation(self, text: str) -> Dict[str, Any]:
        """Test 4: Directly modify newline embedding."""
        print("  🧪 Test 4: Embedding Manipulation")
        
        try:
            # Original generation
            original_prompt = text[:100]
            original_gen = self.generate_text(original_prompt)
            original_cycles = count_cycles_standardized(original_gen, self.tokenizer)
            
            # Test 1: Zero out newline embedding
            self.model.gpt_neox.embed_in.weight[self.newline_token_id] = torch.zeros_like(self.original_embedding)
            zero_gen = self.generate_text(original_prompt)
            zero_cycles = count_cycles_standardized(zero_gen, self.tokenizer)
            
            # Test 2: Random embedding
            self.model.gpt_neox.embed_in.weight[self.newline_token_id] = torch.randn_like(self.original_embedding)
            random_gen = self.generate_text(original_prompt)
            random_cycles = count_cycles_standardized(random_gen, self.tokenizer)
            
            # Test 3: Amplified original embedding (2x)
            self.model.gpt_neox.embed_in.weight[self.newline_token_id] = self.original_embedding * 2.0
            amplified_gen = self.generate_text(original_prompt)
            amplified_cycles = count_cycles_standardized(amplified_gen, self.tokenizer)
            
            # Restore original
            self.model.gpt_neox.embed_in.weight[self.newline_token_id] = self.original_embedding.clone()
            
            return {
                'original_cycles': original_cycles,
                'zero_embedding_cycles': zero_cycles,
                'random_embedding_cycles': random_cycles,
                'amplified_embedding_cycles': amplified_cycles,
                'zero_reduces_repetition': zero_cycles < original_cycles,
                'amplification_increases_repetition': amplified_cycles > original_cycles,
                'amplified_sample': amplified_gen[:150]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_attention_blocking(self, text: str) -> Dict[str, Any]:
        """Test 5: Block attention to/from newlines."""
        print("  🧪 Test 5: Attention Blocking")
        
        original_prompt = text[:100]
        
        # This is complex to implement correctly, so let's do a simpler version
        # by masking newline positions in attention
        
        def attention_masking_hook(module, input, output):
            # output[0] is attention weights
            if len(output) > 0 and hasattr(output[0], 'shape'):
                attn_weights = output[0]
                if len(attn_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    # Find newline positions in current sequence
                    current_input = input[0] if isinstance(input[0], torch.Tensor) else None
                    if current_input is not None and len(current_input.shape) >= 2:
                        newline_mask = (current_input == self.newline_token_id)
                        if newline_mask.any():
                            # Zero out attention to/from newlines
                            for pos in torch.nonzero(newline_mask, as_tuple=True)[1]:
                                if pos < attn_weights.size(-1):
                                    attn_weights[:, :, :, pos] = 0  # No attention TO newline
                                    attn_weights[:, :, pos, :] = 0  # No attention FROM newline
                            
                            # Renormalize
                            attn_weights = F.softmax(attn_weights, dim=-1)
                            output = (attn_weights,) + output[1:]
            return output
        
        try:
            # Install hooks on later layers
            hooks = []
            for layer_idx in [17, 19, 21]:
                if layer_idx < len(self.model.gpt_neox.layers):
                    layer = self.model.gpt_neox.layers[layer_idx]
                    hook = layer.attention.register_forward_hook(attention_masking_hook)
                    hooks.append(hook)
            
            # Generate with blocked attention
            blocked_gen = self.generate_text(original_prompt)
            blocked_cycles = count_cycles_standardized(blocked_gen, self.tokenizer)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Original for comparison
            original_gen = self.generate_text(original_prompt)
            original_cycles = count_cycles_standardized(original_gen, self.tokenizer)
            
            return {
                'original_cycles': original_cycles,
                'blocked_attention_cycles': blocked_cycles,
                'blocking_reduces_repetition': blocked_cycles < original_cycles,
                'blocked_sample': blocked_gen[:150]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_comprehensive_newline_experiments(self, test_texts: List[str]) -> Dict[str, Any]:
        """Run all newline intervention experiments."""
        if not self.load_model():
            return {'error': 'Failed to load model'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'newline_token_id': self.newline_token_id,
            'experiments': [],
            'summary': {}
        }
        
        print("🧪 DIRECT NEWLINE INTERVENTION EXPERIMENTS")
        print("=" * 60)
        
        # Filter for texts with newlines
        texts_with_newlines = [text for text in test_texts if '\n' in text]
        if not texts_with_newlines:
            texts_with_newlines = [
                "Line 1: Introduction\nLine 2: Main content\nLine 3: Details\nLine 4: Conclusion",
                "Item A\nItem B\nItem C\nItem D\nFinal item",
                "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
            ]
        
        for i, text in enumerate(texts_with_newlines[:3]):
            print(f"\n--- Testing Text {i+1}/3 ---")
            print(f"Text sample: {text[:80]}...")
            
            experiment_result = {
                'text_index': i,
                'text_sample': text[:100],
                'tests': {}
            }
            
            # Run all tests
            experiment_result['tests']['newline_removal'] = self.test_newline_removal(text)
            experiment_result['tests']['newline_amplification'] = self.test_newline_amplification(text)
            experiment_result['tests']['token_replacement'] = self.test_token_replacement(text)
            experiment_result['tests']['embedding_manipulation'] = self.test_embedding_manipulation(text)
            experiment_result['tests']['attention_blocking'] = self.test_attention_blocking(text)
            
            results['experiments'].append(experiment_result)
            
            # Print immediate results
            removal = experiment_result['tests']['newline_removal']
            amplification = experiment_result['tests']['newline_amplification']
            
            print(f"  Removal: {removal['original_cycles']} → {removal['no_newline_cycles']} cycles ({'✅' if removal['causal_evidence'] else '❌'})")
            print(f"  Amplification: {amplification['original_cycles']} → {amplification['triple_cycles']} cycles ({'✅' if amplification['amplification_effect'] else '❌'})")
        
        # Generate summary
        results['summary'] = self.analyze_newline_results(results['experiments'])
        
        return results
    
    def analyze_newline_results(self, experiments: List[Dict]) -> Dict[str, Any]:
        """Analyze results to determine causal evidence."""
        evidence_counts = {
            'removal_reduces': 0,
            'amplification_increases': 0,
            'embedding_manipulation_effective': 0,
            'attention_blocking_effective': 0,
            'total_tests': len(experiments)
        }
        
        strongest_effects = []
        
        for exp in experiments:
            tests = exp['tests']
            
            # Removal evidence
            if tests['newline_removal'].get('causal_evidence', False):
                evidence_counts['removal_reduces'] += 1
            
            # Amplification evidence
            if tests['newline_amplification'].get('amplification_effect', False):
                evidence_counts['amplification_increases'] += 1
            
            # Embedding evidence
            embed_test = tests['embedding_manipulation']
            if (embed_test.get('zero_reduces_repetition', False) or 
                embed_test.get('amplification_increases_repetition', False)):
                evidence_counts['embedding_manipulation_effective'] += 1
            
            # Attention blocking evidence
            if tests['attention_blocking'].get('blocking_reduces_repetition', False):
                evidence_counts['attention_blocking_effective'] += 1
        
        # Calculate evidence strength
        total_evidence = sum([
            evidence_counts['removal_reduces'],
            evidence_counts['amplification_increases'],
            evidence_counts['embedding_manipulation_effective'],
            evidence_counts['attention_blocking_effective']
        ])
        
        max_possible_evidence = evidence_counts['total_tests'] * 4
        evidence_strength = total_evidence / max_possible_evidence * 100
        
        # Determine conclusion
        if evidence_strength >= 60:
            conclusion = "🎯 STRONG CAUSAL EVIDENCE: Newlines causally drive repetition"
            recommendation = "Develop newline-targeted repetition induction techniques"
        elif evidence_strength >= 30:
            conclusion = "📈 MODERATE EVIDENCE: Newlines likely involved in repetition mechanism"  
            recommendation = "Further investigate newline processing mechanisms"
        else:
            conclusion = "❌ WEAK EVIDENCE: Newlines may be correlational not causal"
            recommendation = "Investigate alternative repetition mechanisms"
        
        return {
            'evidence_counts': evidence_counts,
            'evidence_strength_percent': evidence_strength,
            'conclusion': conclusion,
            'recommendation': recommendation,
            'causal_mechanism_confirmed': evidence_strength >= 60
        }


def main():
    """Main experiment function."""
    print("🚀 PHASE 3: DIRECT NEWLINE TOKEN INTERVENTIONS")
    print("=" * 60)
    
    # Load test data
    print("Loading test data...")
    try:
        test_texts = load_cached_dataset("JeanKaddour/minipile", "train", 42, n_samples=5)
        print(f"✅ Loaded {len(test_texts)} test texts from JeanKaddour/minipile")
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        test_texts = [
            "Introduction section here.\nMain content follows.\nDetailed analysis.\nConclusions drawn.",
            "Step 1: Initialize\nStep 2: Process\nStep 3: Validate\nStep 4: Complete",
            "First point made.\n\nSecond point elaborated.\n\nThird point concluded.",
            "Category A items\nCategory B items\nCategory C items\nFinal category",
            "Beginning of story.\nMiddle developments.\nClimax reached.\nResolution achieved."
        ]
        print(f"Using {len(test_texts)} fallback texts")
    
    # Run experiments
    experiment = NewlineInterventionExperiment()
    results = experiment.run_comprehensive_newline_experiments(test_texts)
    
    if 'error' in results:
        print(f"❌ Experiments failed: {results['error']}")
        return False
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 NEWLINE CAUSALITY EXPERIMENT SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"\n{summary['conclusion']}")
    print(f"\nEvidence Strength: {summary['evidence_strength_percent']:.1f}%")
    print(f"Recommendation: {summary['recommendation']}")
    
    print(f"\nDetailed Evidence:")
    for evidence_type, count in summary['evidence_counts'].items():
        if evidence_type != 'total_tests':
            print(f"  {evidence_type}: {count}/{summary['evidence_counts']['total_tests']} tests")
    
    # Save results
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/newline_causality"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'newline_causality_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    if summary['causal_mechanism_confirmed']:
        print(f"\n🎉 BREAKTHROUGH: Causal mechanism identified!")
        print(f"🎯 Next: Develop newline-based repetition induction!")
    else:
        print(f"\n🔍 Continue investigating alternative mechanisms...")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)