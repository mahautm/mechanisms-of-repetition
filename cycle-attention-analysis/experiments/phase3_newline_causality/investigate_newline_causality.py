#!/usr/bin/env python3
"""
Newline Token Investigation: What's Stored in the Newline?

This script investigates what information might be encoded in or triggered by 
the newline token that could be causing (not just correlating with) repetition.
"""

import torch
import numpy as np
import json
import os
import statistics
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
    from parrots.cycle_detection import detect_cycles
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class NewlineInvestigator:
    """Investigates what's encoded in the newline token that causes repetition."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.newline_token_id = None
        
    def load_model(self):
        """Load model and tokenizer."""
        print("Loading model...")
        try:
            self.model, self.tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Find newline token
            self.newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
            print(f"✅ Model loaded. Newline token ID: {self.newline_token_id}")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def analyze_newline_embedding(self) -> Dict[str, Any]:
        """Analyze the embedding vector of the newline token."""
        print("🔍 Analyzing newline token embedding...")
        
        # Get embedding matrix
        embedding_matrix = self.model.gpt_neox.embed_in.weight  # [vocab_size, hidden_dim]
        newline_embedding = embedding_matrix[self.newline_token_id]  # [hidden_dim]
        
        # Compare to other tokens
        common_tokens = {
            'space': self.tokenizer.encode(' ', add_special_tokens=False)[0],
            'period': self.tokenizer.encode('.', add_special_tokens=False)[0],
            'the': self.tokenizer.encode('the', add_special_tokens=False)[0],
            'and': self.tokenizer.encode('and', add_special_tokens=False)[0],
            'a': self.tokenizer.encode('a', add_special_tokens=False)[0],
        }
        
        # Calculate similarities
        similarities = {}
        for name, token_id in common_tokens.items():
            token_embedding = embedding_matrix[token_id]
            similarity = torch.cosine_similarity(newline_embedding, token_embedding, dim=0)
            similarities[name] = float(similarity)
        
        # Analyze embedding statistics
        newline_stats = {
            'norm': float(torch.norm(newline_embedding)),
            'mean': float(torch.mean(newline_embedding)),
            'std': float(torch.std(newline_embedding)),
            'min': float(torch.min(newline_embedding)),
            'max': float(torch.max(newline_embedding)),
            'similarities_to_common_tokens': similarities
        }
        
        # Find most similar tokens in vocabulary
        all_similarities = torch.cosine_similarity(
            newline_embedding.unsqueeze(0), 
            embedding_matrix, 
            dim=1
        )
        top_similar_indices = torch.topk(all_similarities, k=10).indices
        
        most_similar = []
        for idx in top_similar_indices:
            token = self.tokenizer.decode([int(idx)])
            similarity = float(all_similarities[idx])
            most_similar.append({'token': repr(token), 'similarity': similarity})
        
        newline_stats['most_similar_tokens'] = most_similar
        
        return newline_stats
    
    def analyze_newline_attention_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze what the newline token attends to and what attends to it."""
        print("🔍 Analyzing newline attention patterns...")
        
        # Tokenize
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0]]
        
        # Find newline positions
        newline_positions = [i for i, token_id in enumerate(input_ids[0]) 
                           if token_id == self.newline_token_id]
        
        if not newline_positions:
            return {'error': 'No newlines found in text'}
        
        # Get attention weights with hooks
        attention_weights = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                # output[0] is attention weights: (batch, heads, seq_len, seq_len)
                attention_weights[name] = output[0].detach().cpu()
            return hook
        
        # Register hooks on later layers where repetition patterns emerge
        hooks = []
        target_layers = [15, 17, 19, 21]  # Later layers
        for layer_idx in target_layers:
            if layer_idx < len(self.model.gpt_neox.layers):
                layer = self.model.gpt_neox.layers[layer_idx]
                hook = layer.attention.register_forward_hook(attention_hook(f"layer_{layer_idx}"))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze attention patterns for each newline
        newline_analysis = []
        
        for pos in newline_positions:
            analysis = {'position': pos, 'token_context': tokens[max(0, pos-3):pos+4]}
            
            # For each layer, analyze what newline attends to and what attends to newline
            layer_analysis = {}
            for layer_name, attn in attention_weights.items():
                # Check attention tensor dimensions
                print(f"    Attention tensor shape for {layer_name}: {attn.shape}")
                
                # Initialize variables
                newline_attends_to = None
                attends_to_newline = None
                seq_len = 0
                
                # Handle different tensor shapes - sometimes it's [heads, seq_len, seq_len]
                if len(attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    batch_idx, heads, seq_len, _ = attn.shape
                    if pos < seq_len:
                        # What does newline attend to? (average across heads)
                        newline_attends_to = torch.mean(attn[0, :, pos, :], dim=0)  # [seq_len]
                        # What attends to newline? (average across heads)
                        attends_to_newline = torch.mean(attn[0, :, :, pos], dim=0)  # [seq_len]
                elif len(attn.shape) == 3:  # [heads, seq_len, seq_len]
                    heads, seq_len, _ = attn.shape
                    if pos < seq_len:
                        # What does newline attend to? (average across heads)
                        newline_attends_to = torch.mean(attn[:, pos, :], dim=0)  # [seq_len]
                        # What attends to newline? (average across heads)
                        attends_to_newline = torch.mean(attn[:, :, pos], dim=0)  # [seq_len]
                else:
                    print(f"    Unexpected attention tensor shape: {attn.shape}")
                    continue
                
                # Process attention results if we have valid data
                if pos < seq_len and newline_attends_to is not None and attends_to_newline is not None:
                    # Top attended tokens by newline
                    top_attended = torch.topk(newline_attends_to, k=5)
                    newline_attends_list = []
                    for i, score in zip(top_attended.indices, top_attended.values):
                        if i < len(tokens):
                            newline_attends_list.append({
                                'position': int(i),
                                'token': repr(tokens[int(i)]),
                                'attention_score': float(score)
                            })
                    
                    # Top tokens attending to newline
                    top_attending = torch.topk(attends_to_newline, k=5)
                    attending_to_newline_list = []
                    for i, score in zip(top_attending.indices, top_attending.values):
                        if i < len(tokens):
                            attending_to_newline_list.append({
                                'position': int(i),
                                'token': repr(tokens[int(i)]),
                                'attention_score': float(score)
                            })
                    
                    layer_analysis[layer_name] = {
                        'newline_attends_to': newline_attends_list,
                        'attends_to_newline': attending_to_newline_list,
                        'self_attention_score': float(newline_attends_to[pos])
                    }
            
            analysis['layer_analysis'] = layer_analysis
            newline_analysis.append(analysis)
        
        return {
            'text_length': len(tokens),
            'newline_count': len(newline_positions),
            'newline_positions': newline_positions,
            'newline_analysis': newline_analysis
        }
    
    def test_newline_intervention_mechanisms(self, text: str) -> Dict[str, Any]:
        """Test different mechanisms by which newline might cause repetition."""
        print("🔍 Testing newline intervention mechanisms...")
        
        results = {}
        
        # 1. Test newline removal - does removing newlines prevent repetition?
        text_no_newlines = text.replace('\n', ' ')
        baseline_gen = self.generate_text(text[:100])
        no_newline_gen = self.generate_text(text_no_newlines[:100])
        
        baseline_cycles = detect_cycles(baseline_gen)
        no_newline_cycles = detect_cycles(no_newline_gen)
        
        results['newline_removal_test'] = {
            'baseline_cycles': len(baseline_cycles) if baseline_cycles else 0,
            'no_newline_cycles': len(no_newline_cycles) if no_newline_cycles else 0,
            'repetition_reduced': (len(baseline_cycles) if baseline_cycles else 0) > (len(no_newline_cycles) if no_newline_cycles else 0)
        }
        
        # 2. Test newline replacement - does replacing with other tokens change behavior?
        replacements = {
            'space': text.replace('\n', ' '),
            'period': text.replace('\n', '.'),
            'semicolon': text.replace('\n', ';'),
            'double_space': text.replace('\n', '  ')
        }
        
        replacement_results = {}
        for replacement_name, modified_text in replacements.items():
            gen_text = self.generate_text(modified_text[:100])
            cycles = detect_cycles(gen_text)
            replacement_results[replacement_name] = {
                'cycles': len(cycles) if cycles else 0,
                'generated_text_sample': gen_text[:200]
            }
        
        results['replacement_tests'] = replacement_results
        
        # 3. Test newline amplification - does adding more newlines increase repetition?
        text_extra_newlines = text.replace('\n', '\n\n')
        text_many_newlines = text.replace('\n', '\n\n\n')
        
        extra_gen = self.generate_text(text_extra_newlines[:100])
        many_gen = self.generate_text(text_many_newlines[:100])
        
        extra_cycles = detect_cycles(extra_gen)
        many_cycles = detect_cycles(many_gen)
        
        results['amplification_test'] = {
            'baseline_cycles': len(baseline_cycles) if baseline_cycles else 0,
            'extra_newline_cycles': len(extra_cycles) if extra_cycles else 0,
            'many_newline_cycles': len(many_cycles) if many_cycles else 0,
            'amplification_effect': (len(many_cycles) if many_cycles else 0) > (len(baseline_cycles) if baseline_cycles else 0)
        }
        
        return results
    
    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate text from prompt."""
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
    
    def analyze_newline_hidden_states(self, text: str) -> Dict[str, Any]:
        """Analyze hidden state evolution through layers for newline tokens."""
        print("🔍 Analyzing newline hidden state evolution...")
        
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        newline_positions = [i for i, token_id in enumerate(input_ids[0]) 
                           if token_id == self.newline_token_id]
        
        if not newline_positions:
            return {'error': 'No newlines found'}
        
        # Store hidden states from all layers
        hidden_states = {}
        
        def hidden_state_hook(name):
            def hook(module, input, output):
                # output is the hidden state: (batch, seq_len, hidden_dim)
                if isinstance(output, tuple):
                    hidden_states[name] = output[0].detach().cpu()
                else:
                    hidden_states[name] = output.detach().cpu()
            return hook
        
        # Register hooks
        hooks = []
        for layer_idx in range(len(self.model.gpt_neox.layers)):
            layer = self.model.gpt_neox.layers[layer_idx]
            hook = layer.register_forward_hook(hidden_state_hook(f"layer_{layer_idx}"))
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze hidden state patterns for newlines
        newline_hidden_analysis = []
        
        for pos in newline_positions[:3]:  # Analyze first 3 newlines
            analysis = {'position': pos}
            
            # Track how newline representation changes through layers
            layer_representations = {}
            layer_norms = {}
            layer_changes = {}
            
            prev_hidden = None
            for layer_name in sorted(hidden_states.keys(), key=lambda x: int(x.split('_')[1])):
                if layer_name in hidden_states:
                    hidden = hidden_states[layer_name]
                    if pos < hidden.size(1):
                        newline_vector = hidden[0, pos, :]  # [hidden_dim]
                        
                        layer_representations[layer_name] = {
                            'norm': float(torch.norm(newline_vector)),
                            'mean': float(torch.mean(newline_vector)),
                            'std': float(torch.std(newline_vector))
                        }
                        layer_norms[layer_name] = float(torch.norm(newline_vector))
                        
                        # Calculate change from previous layer
                        if prev_hidden is not None and pos < prev_hidden.size(1):
                            prev_vector = prev_hidden[0, pos, :]
                            change_magnitude = float(torch.norm(newline_vector - prev_vector))
                            cosine_sim = float(torch.cosine_similarity(
                                newline_vector, prev_vector, dim=0
                            ))
                            layer_changes[layer_name] = {
                                'change_magnitude': change_magnitude,
                                'cosine_similarity': cosine_sim
                            }
                        
                        prev_hidden = hidden
            
            analysis['layer_representations'] = layer_representations
            analysis['layer_norms'] = layer_norms
            analysis['layer_changes'] = layer_changes
            
            newline_hidden_analysis.append(analysis)
        
        return {
            'newline_count': len(newline_positions),
            'analyzed_positions': newline_positions[:3],
            'newline_hidden_analysis': newline_hidden_analysis
        }
    
    def run_comprehensive_newline_investigation(self, test_texts: List[str]) -> Dict[str, Any]:
        """Run comprehensive investigation of newline token behavior."""
        if not self.load_model():
            return {'error': 'Failed to load model'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'newline_token_id': self.newline_token_id,
            'embedding_analysis': {},
            'attention_analysis': [],
            'intervention_analysis': [],
            'hidden_state_analysis': [],
            'summary': {}
        }
        
        print("🔬 COMPREHENSIVE NEWLINE INVESTIGATION")
        print("=" * 60)
        
        # 1. Analyze embedding properties
        results['embedding_analysis'] = self.analyze_newline_embedding()
        
        # 2. Large-scale batch intervention analysis
        print(f"\n🔄 BATCH INTERVENTION ANALYSIS (1000 samples)")
        print("=" * 60)
        
        all_batch_results = []
        total_batches = (len(test_texts) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(test_texts))
            batch_texts = test_texts[start_idx:end_idx]
            
            batch_results = self.process_batch_interventions(batch_texts, batch_num + 1)
            all_batch_results.append(batch_results)
            
            # Progress update
            processed = end_idx
            print(f"    Progress: {processed}/{len(test_texts)} texts ({processed/len(test_texts)*100:.1f}%)")
        
        # 3. Statistical analysis of results
        print(f"\n📊 STATISTICAL ANALYSIS")
        print("=" * 60)
        
        # Aggregate all results
        all_baseline = []
        all_no_newline = []
        all_improvements = []
        total_with_newlines = 0
        
        for batch in all_batch_results:
            all_baseline.extend(batch['baseline_cycles'])
            all_no_newline.extend(batch['no_newline_cycles'])
            all_improvements.extend(batch['improvements'])
            total_with_newlines += batch['texts_with_newlines']
        
        # Store statistical results
        if all_baseline:
            import statistics
            results['statistical_analysis'] = {
                'total_texts_analyzed': len(test_texts),
                'texts_with_newlines': total_with_newlines,
                'baseline_cycles': {
                    'mean': statistics.mean(all_baseline),
                    'median': statistics.median(all_baseline),
                    'std': statistics.stdev(all_baseline) if len(all_baseline) > 1 else 0,
                    'min': min(all_baseline),
                    'max': max(all_baseline)
                },
                'no_newline_cycles': {
                    'mean': statistics.mean(all_no_newline),
                    'median': statistics.median(all_no_newline),
                    'std': statistics.stdev(all_no_newline) if len(all_no_newline) > 1 else 0,
                    'min': min(all_no_newline),
                    'max': max(all_no_newline)
                },
                'improvements': {
                    'mean': statistics.mean(all_improvements),
                    'median': statistics.median(all_improvements),
                    'std': statistics.stdev(all_improvements) if len(all_improvements) > 1 else 0,
                    'positive_improvements': sum(1 for x in all_improvements if x > 0),
                    'negative_improvements': sum(1 for x in all_improvements if x < 0),
                    'no_change': sum(1 for x in all_improvements if x == 0)
                }
            }
            
            # Print statistical summary
            stats = results['statistical_analysis']
            print(f"  📈 Baseline cycles: {stats['baseline_cycles']['mean']:.2f} ± {stats['baseline_cycles']['std']:.2f}")
            print(f"  📉 No-newline cycles: {stats['no_newline_cycles']['mean']:.2f} ± {stats['no_newline_cycles']['std']:.2f}")
            print(f"  🎯 Improvement: {stats['improvements']['mean']:.2f} ± {stats['improvements']['std']:.2f}")
            print(f"  ✅ Texts improved: {stats['improvements']['positive_improvements']}/{total_with_newlines} ({stats['improvements']['positive_improvements']/total_with_newlines*100:.1f}%)")
            print(f"  ❌ Texts worsened: {stats['improvements']['negative_improvements']}/{total_with_newlines} ({stats['improvements']['negative_improvements']/total_with_newlines*100:.1f}%)")
            print(f"  ➡️ No change: {stats['improvements']['no_change']}/{total_with_newlines} ({stats['improvements']['no_change']/total_with_newlines*100:.1f}%)")
        
        # Store batch results for detailed analysis
        results['batch_results'] = all_batch_results
        
        # 3. Generate summary insights
        results['summary'] = self.generate_investigation_summary(results)
        
        return results
    
    def generate_investigation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary insights from investigation."""
        summary = {
            'key_findings': [],
            'causal_evidence': {},
            'mechanistic_insights': [],
            'recommendations': []
        }
        
        # Analyze embedding uniqueness
        embedding = results['embedding_analysis']
        if 'similarities_to_common_tokens' in embedding:
            max_similarity = max(embedding['similarities_to_common_tokens'].values())
            if max_similarity < 0.5:
                summary['key_findings'].append(
                    f"🔍 Newline embedding is highly unique (max similarity to common tokens: {max_similarity:.3f})"
                )
        
        # Analyze intervention evidence
        intervention_effects = []
        for item in results['intervention_analysis']:
            analysis = item['analysis']
            if 'newline_removal_test' in analysis:
                test = analysis['newline_removal_test']
                if test['repetition_reduced']:
                    intervention_effects.append('removal_reduces_repetition')
            
            if 'amplification_test' in analysis:
                test = analysis['amplification_test']
                if test['amplification_effect']:
                    intervention_effects.append('amplification_increases_repetition')
        
        if intervention_effects:
            summary['causal_evidence']['intervention_effects'] = intervention_effects
            if 'removal_reduces_repetition' in intervention_effects:
                summary['key_findings'].append(
                    "⚡ CAUSAL EVIDENCE: Removing newlines reduces repetition"
                )
            if 'amplification_increases_repetition' in intervention_effects:
                summary['key_findings'].append(
                    "⚡ CAUSAL EVIDENCE: Adding more newlines increases repetition"
                )
        
        # Generate recommendations
        if len(summary['key_findings']) > 0:
            summary['recommendations'].extend([
                "🎯 Focus interventions on newline token processing",
                "🔧 Test newline token replacement in vocabulary",
                "📊 Investigate newline-specific attention heads",
                "⚡ Develop newline-targeted repetition induction"
            ])
        else:
            summary['recommendations'].extend([
                "🔍 Investigate other structural tokens",
                "📈 Look for alternative repetition mechanisms",
                "🔬 Test different model architectures"
            ])
        
        return summary


def main():
    """Main investigation function."""
    print("🔬 NEWLINE TOKEN CAUSAL INVESTIGATION")
    print("=" * 60)
    
    # Load test data with newlines
    print("Loading test data with newlines...")
    try:
        # Load test data with newlines - use cached loader for efficiency
        test_texts = load_cached_dataset("JeanKaddour/minipile", "train", 42, n_samples=1000)
        # Filter for texts with newlines
        texts_with_newlines = [text for text in test_texts if '\n' in text]
        if not texts_with_newlines:
            print("⚠️ No texts with newlines found, using fallback")
            texts_with_newlines = [
                "Line 1 content\nLine 2 content\nLine 3 content\nRepeat pattern",
                "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
                "Item 1\nItem 2\nItem 3\nItem 4\nFinal item"
            ]
        
        print(f"✅ Found {len(texts_with_newlines)} texts with newlines")
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        texts_with_newlines = [
            "Line 1 content\nLine 2 content\nLine 3 content",
            "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            "Item 1\nItem 2\nItem 3\nItem 4"
        ]
        print(f"Using {len(texts_with_newlines)} fallback texts")
    
    # Truncate texts to avoid memory issues while preserving newlines
    max_chars = 500  # Character-based limit to preserve newlines
    truncated_texts = []
    for text in texts_with_newlines:
        if len(text) > max_chars:
            # Find good cut point near character limit that preserves structure
            cutoff = text[:max_chars].rfind('\n')
            if cutoff == -1:  # No newline found, cut at word boundary
                cutoff = text[:max_chars].rfind(' ')
                if cutoff == -1:  # No space found, hard cut
                    cutoff = max_chars
            truncated_text = text[:cutoff]
            truncated_texts.append(truncated_text)
        else:
            truncated_texts.append(text)
    
    print(f"✅ Truncated to max {max_chars} chars for memory efficiency")
    
    # Debug: Check if newlines are preserved
    for i, text in enumerate(truncated_texts):
        newline_count = text.count('\n')
        print(f"    Text {i+1}: {len(text)} chars, {newline_count} newlines")

    # Run investigation
    investigator = NewlineInvestigator()
    results = investigator.run_comprehensive_newline_investigation(truncated_texts)
    
    if 'error' in results:
        print(f"❌ Investigation failed: {results['error']}")
        return False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 NEWLINE INVESTIGATION SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    
    print("\n🔍 KEY FINDINGS:")
    for finding in summary['key_findings']:
        print(f"  {finding}")
    
    if 'causal_evidence' in summary and summary['causal_evidence']:
        print(f"\n⚡ CAUSAL EVIDENCE DETECTED:")
        for evidence_type, evidence in summary['causal_evidence'].items():
            print(f"  {evidence_type}: {evidence}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    
    # Save results
    output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/newline_investigation"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'newline_causal_investigation.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)