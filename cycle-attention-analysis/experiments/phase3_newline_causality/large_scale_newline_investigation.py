#!/usr/bin/env python3
"""
Large-Scale Newline Investigation: 1000+ Samples

Statistical analysis of newline token causality with robust sample size.
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

try:
    from datasets import load_dataset
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    from parrots.cycle_detection import detect_cycles
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class LargeScaleNewlineInvestigator:
    """Large-scale statistical investigation of newline causality."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "EleutherAI/pythia-1.4b"
        self.batch_size = 50  # Process 50 texts at a time
        self.cleanup_frequency = 10  # Clean GPU memory every 10 batches
        
        # Load model and tokenizer
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = GPTNeoXForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        # Get newline token ID
        self.newline_token_id = self.tokenizer.encode('\n')[0]
        print(f"✅ Model loaded. Newline token ID: {self.newline_token_id}")
        
        # Initialize detection function
        self.detect_cycles = detect_cycles

    def load_large_dataset(self, n_samples: int = 1000) -> List[str]:
        """Load n_samples from JeanKaddour/minipile with newlines."""
        print(f"Loading {n_samples} samples from JeanKaddour/minipile...")
        
        # Load dataset
        dataset = load_dataset("JeanKaddour/minipile", split="train")
        
        # Find texts with newlines
        texts_with_newlines = []
        max_chars = 300  # Keep reasonable length for processing
        
        for i, example in enumerate(dataset):
            if i >= 10000:  # Don't search entire dataset
                break
                
            text = example['text']
            if '\n' in text and len(text) > 50:  # Must have newlines and minimum length
                # Truncate to manageable size while preserving newlines
                if len(text) > max_chars:
                    truncated = text[:max_chars]
                    # Try to end at a reasonable point
                    if '\n' in truncated:
                        texts_with_newlines.append(truncated)
                else:
                    texts_with_newlines.append(text)
                
                if len(texts_with_newlines) >= n_samples:
                    break
        
        print(f"✅ Found {len(texts_with_newlines)} texts with newlines")
        
        # Statistics on dataset
        newline_counts = [text.count('\n') for text in texts_with_newlines]
        avg_newlines = sum(newline_counts) / len(newline_counts)
        avg_length = sum(len(text) for text in texts_with_newlines) / len(texts_with_newlines)
        
        print(f"   • Average newlines per text: {avg_newlines:.1f}")
        print(f"   • Average text length: {avg_length:.1f} chars")
        
        return texts_with_newlines

    def process_batch_interventions(self, texts_batch: List[str], batch_num: int) -> Dict:
        """Process a batch of texts for intervention analysis."""
        print(f"  📦 Processing batch {batch_num}: {len(texts_batch)} texts")
        
        batch_results = {
            'batch_number': batch_num,
            'size': len(texts_batch),
            'baseline_cycles': [],
            'no_newline_cycles': [],
            'improvements': [],
            'texts_with_newlines': 0,
            'processing_errors': 0
        }
        
        for i, text in enumerate(texts_batch):
            # Skip texts without newlines
            if '\n' not in text:
                continue
                
            batch_results['texts_with_newlines'] += 1
            
            try:
                # Baseline: count cycles with newlines
                baseline_result = self.detect_cycles(text)
                # Extract just the cycle count (detect_cycles returns a tuple)
                baseline_cycles = baseline_result[1] if isinstance(baseline_result, tuple) else baseline_result
                batch_results['baseline_cycles'].append(baseline_cycles)
                
                # Intervention: replace newlines with spaces
                no_newline_text = text.replace('\n', ' ')
                no_newline_result = self.detect_cycles(no_newline_text)
                # Extract just the cycle count
                no_newline_cycles = no_newline_result[1] if isinstance(no_newline_result, tuple) else no_newline_result
                batch_results['no_newline_cycles'].append(no_newline_cycles)
                
                # Calculate improvement (positive = fewer cycles after removal)
                improvement = baseline_cycles - no_newline_cycles
                batch_results['improvements'].append(improvement)
                
            except Exception as e:
                batch_results['processing_errors'] += 1
                print(f"    ⚠️ Error processing text {i}: {e}")
                continue
        
        # Memory cleanup
        if batch_num % self.cleanup_frequency == 0:
            torch.cuda.empty_cache()
            
        return batch_results

    def run_large_scale_analysis(self, n_samples: int = 1000) -> Dict:
        """Run comprehensive large-scale newline causality analysis."""
        print("🔬 LARGE-SCALE NEWLINE CAUSALITY INVESTIGATION")
        print("=" * 60)
        
        # Load dataset
        texts = self.load_large_dataset(n_samples)
        
        # Process in batches
        print(f"\n🔄 BATCH PROCESSING ({self.batch_size} texts per batch)")
        print("-" * 60)
        
        all_batch_results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            batch_results = self.process_batch_interventions(batch_texts, batch_num + 1)
            all_batch_results.append(batch_results)
            
            # Progress update
            processed = end_idx
            progress_pct = processed / len(texts) * 100
            print(f"    Progress: {processed}/{len(texts)} texts ({progress_pct:.1f}%)")
        
        # Aggregate statistical results
        print(f"\n📊 STATISTICAL ANALYSIS")
        print("=" * 60)
        
        # Collect all results
        all_baseline = []
        all_no_newline = []
        all_improvements = []
        total_with_newlines = 0
        total_errors = 0
        
        for batch in all_batch_results:
            all_baseline.extend(batch['baseline_cycles'])
            all_no_newline.extend(batch['no_newline_cycles'])
            all_improvements.extend(batch['improvements'])
            total_with_newlines += batch['texts_with_newlines']
            total_errors += batch['processing_errors']
        
        # Calculate statistics
        results = {
            'timestamp': datetime.now().isoformat(),
            'experiment_params': {
                'n_samples_requested': n_samples,
                'n_samples_loaded': len(texts),
                'n_samples_with_newlines': total_with_newlines,
                'processing_errors': total_errors,
                'batch_size': self.batch_size
            },
            'batch_results': all_batch_results
        }
        
        if all_baseline:
            # Statistical analysis
            stats = {
                'sample_size': len(all_baseline),
                'baseline_cycles': {
                    'mean': statistics.mean(all_baseline),
                    'median': statistics.median(all_baseline),
                    'std': statistics.stdev(all_baseline) if len(all_baseline) > 1 else 0,
                    'min': min(all_baseline),
                    'max': max(all_baseline),
                    'distribution': self._calculate_distribution(all_baseline)
                },
                'no_newline_cycles': {
                    'mean': statistics.mean(all_no_newline),
                    'median': statistics.median(all_no_newline),
                    'std': statistics.stdev(all_no_newline) if len(all_no_newline) > 1 else 0,
                    'min': min(all_no_newline),
                    'max': max(all_no_newline),
                    'distribution': self._calculate_distribution(all_no_newline)
                },
                'improvements': {
                    'mean': statistics.mean(all_improvements),
                    'median': statistics.median(all_improvements),
                    'std': statistics.stdev(all_improvements) if len(all_improvements) > 1 else 0,
                    'min': min(all_improvements),
                    'max': max(all_improvements),
                    'positive_count': sum(1 for x in all_improvements if x > 0),
                    'negative_count': sum(1 for x in all_improvements if x < 0),
                    'zero_count': sum(1 for x in all_improvements if x == 0),
                    'distribution': self._calculate_distribution(all_improvements)
                }
            }
            
            results['statistical_analysis'] = stats
            
            # Print summary
            print(f"  📈 Sample size: {stats['sample_size']} texts")
            print(f"  📊 Baseline cycles: {stats['baseline_cycles']['mean']:.2f} ± {stats['baseline_cycles']['std']:.2f}")
            print(f"  📉 No-newline cycles: {stats['no_newline_cycles']['mean']:.2f} ± {stats['no_newline_cycles']['std']:.2f}")
            print(f"  🎯 Mean improvement: {stats['improvements']['mean']:.3f} ± {stats['improvements']['std']:.3f}")
            print()
            print(f"  ✅ Texts improved: {stats['improvements']['positive_count']}/{stats['sample_size']} ({stats['improvements']['positive_count']/stats['sample_size']*100:.1f}%)")
            print(f"  ❌ Texts worsened: {stats['improvements']['negative_count']}/{stats['sample_size']} ({stats['improvements']['negative_count']/stats['sample_size']*100:.1f}%)")
            print(f"  ➡️ No change: {stats['improvements']['zero_count']}/{stats['sample_size']} ({stats['improvements']['zero_count']/stats['sample_size']*100:.1f}%)")
            
            # Statistical significance assessment
            improvement_rate = stats['improvements']['positive_count'] / stats['sample_size']
            mean_effect = stats['improvements']['mean']
            
            print(f"\n🔍 STATISTICAL CONCLUSIONS:")
            if improvement_rate > 0.6:
                print(f"  🟢 STRONG EVIDENCE: {improvement_rate*100:.1f}% improvement rate suggests newlines cause repetition")
            elif improvement_rate > 0.4:
                print(f"  🟡 MODERATE EVIDENCE: {improvement_rate*100:.1f}% improvement rate suggests weak newline causality")  
            else:
                print(f"  🔴 NO EVIDENCE: {improvement_rate*100:.1f}% improvement rate suggests newlines do NOT cause repetition")
                
            if abs(mean_effect) < 0.1:
                print(f"  📊 Effect size is minimal: {mean_effect:.3f} cycles on average")
            elif mean_effect > 0.5:
                print(f"  📈 Substantial positive effect: {mean_effect:.3f} cycles reduced on average")
            elif mean_effect < -0.5:
                print(f"  📉 Substantial negative effect: newlines reduce cycles by {abs(mean_effect):.3f} on average")
        
        # Generate final conclusions
        if 'statistical_analysis' in results:
            stats = results['statistical_analysis']
            conclusions = self._generate_conclusions(stats)
            results['conclusions'] = conclusions
        
        return results
    
    def _calculate_distribution(self, values: List[float]) -> Dict:
        """Calculate distribution statistics."""
        if not values:
            return {}
            
        # Count occurrences
        from collections import Counter
        counts = Counter(values)
        
        return {
            'unique_values': len(counts),
            'most_common': counts.most_common(5),
            'zero_percentage': (values.count(0) / len(values)) * 100,
            'positive_percentage': (sum(1 for x in values if x > 0) / len(values)) * 100,
            'negative_percentage': (sum(1 for x in values if x < 0) / len(values)) * 100
        }
    
    def _generate_conclusions(self, stats: Dict) -> Dict:
        """Generate scientific conclusions from statistical analysis."""
        sample_size = stats['sample_size']
        improvement_rate = stats['improvements']['positive_count'] / sample_size
        mean_effect = stats['improvements']['mean']
        
        # Confidence based on sample size
        if sample_size >= 1000:
            confidence = "High"
        elif sample_size >= 500:
            confidence = "Moderate-High"
        elif sample_size >= 100:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        # Effect magnitude
        if abs(mean_effect) < 0.05:
            magnitude = "Negligible"
        elif abs(mean_effect) < 0.2:
            magnitude = "Small"
        elif abs(mean_effect) < 0.5:
            magnitude = "Moderate"
        else:
            magnitude = "Large"
        
        # Overall conclusion
        if improvement_rate > 0.6 and mean_effect > 0.1:
            overall = "Newlines appear to cause repetitive cycles"
        elif improvement_rate < 0.4 and abs(mean_effect) < 0.1:
            overall = "No evidence that newlines cause repetitive cycles"
        else:
            overall = "Weak or inconsistent evidence for newline causality"
            
        return {
            'sample_size': sample_size,
            'confidence_level': confidence,
            'effect_magnitude': magnitude,
            'improvement_rate': f"{improvement_rate*100:.1f}%",
            'mean_effect_size': f"{mean_effect:.3f}",
            'overall_conclusion': overall,
            'recommendation': self._get_recommendation(improvement_rate, mean_effect, sample_size)
        }
    
    def _get_recommendation(self, improvement_rate: float, mean_effect: float, sample_size: int) -> str:
        """Get scientific recommendation based on results."""
        if sample_size >= 1000:
            if improvement_rate > 0.6:
                return "PRIORITIZE: Newline removal shows strong statistical benefit"
            elif improvement_rate < 0.4:
                return "DEPRIORITIZE: Newline removal shows no statistical benefit"
            else:
                return "INVESTIGATE: Results are mixed, need deeper analysis"
        else:
            return "EXPAND: Need larger sample size for definitive conclusions"


def main():
    """Run large-scale newline causality investigation."""
    try:
        print("🔬 LARGE-SCALE NEWLINE TOKEN CAUSAL INVESTIGATION")
        print("=" * 60)
        
        # Initialize investigator
        investigator = LargeScaleNewlineInvestigator()
        
        # Run analysis with 1000+ samples
        results = investigator.run_large_scale_analysis(n_samples=1000)
        
        # Save detailed results
        output_dir = "/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/newline_investigation"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "large_scale_newline_investigation.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 RESULTS SAVED")
        print("=" * 60)
        print(f"📁 Full results: {output_file}")
        
        # Print final conclusions
        if 'conclusions' in results:
            conclusions = results['conclusions']
            print(f"\n🎯 FINAL CONCLUSIONS")
            print("=" * 60)
            print(f"  📊 Sample size: {conclusions['sample_size']} texts")
            print(f"  🎯 Confidence: {conclusions['confidence_level']}")
            print(f"  📏 Effect size: {conclusions['effect_magnitude']}")
            print(f"  📈 Improvement rate: {conclusions['improvement_rate']}")
            print(f"  🔍 Mean effect: {conclusions['mean_effect_size']} cycles")
            print(f"  🏁 Conclusion: {conclusions['overall_conclusion']}")
            print(f"  💡 Recommendation: {conclusions['recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)