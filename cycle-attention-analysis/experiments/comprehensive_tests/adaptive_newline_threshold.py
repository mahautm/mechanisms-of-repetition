#!/usr/bin/env python3
"""
Adaptive Multi-Head Newline Focus Experiment
Progressively increases the number of heads focusing on NEWLINE tokens until repetition is detected.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
print("✅ Basic imports done")

from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from modules.cached_data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles

print("✅ All imports successful!")

class AdaptiveNewlineFocusProcessor:
    """Implements progressive newline focus until repetition is detected."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Get number of attention heads per layer
        if hasattr(self.model.config, 'num_attention_heads'):
            self.num_heads = self.model.config.num_attention_heads
        elif hasattr(self.model.config, 'n_head'):
            self.num_heads = self.model.config.n_head
        else:
            self.num_heads = 12  # Default fallback
        
        print(f"🔍 Model has {self.num_heads} attention heads per layer")
        
        # Find newline token
        newline_tokens = []
        for token in ['\n', 'Ċ', '<|newline|>', '\r\n']:
            if token in self.tokenizer.get_vocab():
                newline_tokens.append(self.tokenizer.encode(token)[0])
        
        if newline_tokens:
            self.newline_token_id = newline_tokens[0]
            print(f"🎯 Using newline token: {self.tokenizer.decode([self.newline_token_id])!r} (ID: {self.newline_token_id})")
        else:
            self.newline_token_id = self.tokenizer.eos_token_id
            print(f"⚠️ No newline token found, using EOS: {self.tokenizer.decode([self.newline_token_id])!r}")
    
    def find_newline_positions(self, input_ids):
        """Find positions of newline tokens in input sequence."""
        positions = []
        for batch_idx, sequence in enumerate(input_ids):
            batch_positions = []
            for pos, token_id in enumerate(sequence):
                if token_id == self.newline_token_id:
                    batch_positions.append(pos)
            positions.append(batch_positions)
        return positions
    
    def create_progressive_newline_hook(self, target_layer, target_heads, newline_positions, 
                                       intervention_strength=2.0, focus_multiplier=3.0):
        """Create a hook that forces specified heads to focus on newline tokens."""
        
        def progressive_newline_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, attention_weights = output
            else:
                return output
                
            if attention_weights is None:
                return output
                
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Apply intervention to specified heads
            for batch_idx in range(batch_size):
                if batch_idx < len(newline_positions) and newline_positions[batch_idx]:
                    nl_positions = newline_positions[batch_idx]
                    
                    for head_idx in target_heads:
                        if head_idx < num_heads:
                            # Boost attention to all newline positions
                            for nl_pos in nl_positions:
                                if nl_pos < seq_len:
                                    # All query positions attend more to this newline
                                    attention_weights[batch_idx, head_idx, :, nl_pos] *= focus_multiplier
                            
                            # Also boost attention FROM newline positions to other newlines
                            for nl_pos in nl_positions:
                                if nl_pos < seq_len:
                                    for other_nl_pos in nl_positions:
                                        if other_nl_pos < seq_len:
                                            attention_weights[batch_idx, head_idx, nl_pos, other_nl_pos] *= focus_multiplier
                            
                            # Renormalize attention weights for this head
                            attention_weights[batch_idx, head_idx] = F.softmax(
                                attention_weights[batch_idx, head_idx], dim=-1
                            )
            
            return (hidden_states, attention_weights)
        
        return progressive_newline_hook
    
    def apply_progressive_newline_intervention(self, target_layer, target_heads, newline_positions, 
                                            intervention_strength=2.0, focus_multiplier=3.0):
        """Apply progressive newline intervention to specific layer."""
        
        if hasattr(self.model, 'gpt_neox'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        target_layer_module = layers[target_layer]
        
        if hasattr(target_layer_module, 'attention'):
            attention_module = target_layer_module.attention
        elif hasattr(target_layer_module, 'attn'):
            attention_module = target_layer_module.attn
        elif hasattr(target_layer_module, 'self_attn'):
            attention_module = target_layer_module.self_attn
        else:
            raise ValueError("Cannot find attention module in layer")
        
        hook = self.create_progressive_newline_hook(
            target_layer, target_heads, newline_positions, intervention_strength, focus_multiplier
        )
        handle = attention_module.register_forward_hook(hook)
        
        return handle
    
    def generate_with_progressive_intervention(self, input_ids, target_layer, target_heads, 
                                             max_new_tokens=150, intervention_strength=2.0, 
                                             focus_multiplier=3.0):
        """Generate text with progressive newline intervention."""
        
        # Find newline positions in input
        newline_positions = self.find_newline_positions(input_ids)
        
        # Apply intervention
        hook_handle = self.apply_progressive_newline_intervention(
            target_layer, target_heads, newline_positions, intervention_strength, focus_multiplier
        )
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,  # Disable repetition penalty to allow natural repetition
                )
        finally:
            hook_handle.remove()
        
        return generated
    
    def find_repetition_threshold(self, texts, target_layer, max_heads=None, max_new_tokens=150,
                                intervention_strength=2.0, focus_multiplier=3.0):
        """Progressively increase heads until repetition is detected."""
        
        if max_heads is None:
            max_heads = self.num_heads
        
        results = {
            'texts': texts,
            'target_layer': target_layer,
            'threshold_results': [],  # For each text: {'text': ..., 'threshold': ..., 'generations': [...]}
            'baseline_cycles': [],
            'intervention_strength': intervention_strength,
            'focus_multiplier': focus_multiplier
        }
        
        print(f"🎯 Finding repetition threshold for layer {target_layer}")
        print(f"📊 Testing up to {max_heads} heads with strength {intervention_strength}, multiplier {focus_multiplier}")
        
        for text_idx, text in enumerate(tqdm(texts, desc="Processing texts")):
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=400)
            input_ids = inputs['input_ids'].to(self.device)
            
            # Check if text has newlines
            newline_positions = self.find_newline_positions(input_ids)
            if not any(newline_positions):
                print(f"   ⚠️ Text {text_idx+1} has no newlines, skipping")
                continue
            
            print(f"\n📝 Text {text_idx+1}: Found {len(newline_positions[0])} newline positions")
            print(f"   Input preview: {text[:100]}...")
            
            # Baseline generation (no intervention)
            with torch.no_grad():
                baseline_output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                )
            
            baseline_text = self.tokenizer.decode(baseline_output[0], skip_special_tokens=False)
            baseline_tokens = self.tokenizer.encode(baseline_text, return_tensors='pt')[0]
            baseline_cycles = detect_cycles(baseline_tokens)
            baseline_has_cycles = baseline_cycles is not None
            
            results['baseline_cycles'].append(baseline_cycles)
            
            print(f"   🔄 Baseline cycles: {'YES' if baseline_has_cycles else 'NO'}")
            
            # Progressive head testing
            text_result = {
                'input_text': text,
                'baseline_text': baseline_text,
                'baseline_cycles': baseline_cycles,
                'threshold_head_count': None,
                'threshold_found': False,
                'progression_results': []
            }
            
            # Test increasing numbers of heads
            for num_heads in range(1, max_heads + 1):
                target_heads = list(range(num_heads))
                
                # Generate with intervention
                intervention_output = self.generate_with_progressive_intervention(
                    input_ids, target_layer, target_heads, max_new_tokens,
                    intervention_strength, focus_multiplier
                )
                
                intervention_text = self.tokenizer.decode(intervention_output[0], skip_special_tokens=False)
                intervention_tokens = self.tokenizer.encode(intervention_text, return_tensors='pt')[0]
                intervention_cycles = detect_cycles(intervention_tokens)
                intervention_has_cycles = intervention_cycles is not None
                
                # Record this step
                step_result = {
                    'num_heads': num_heads,
                    'target_heads': target_heads,
                    'generated_text': intervention_text,
                    'cycles': intervention_cycles,
                    'has_cycles': intervention_has_cycles,
                    'repetition_induced': intervention_has_cycles and not baseline_has_cycles
                }
                text_result['progression_results'].append(step_result)
                
                print(f"      Heads 0-{num_heads-1}: Cycles={'YES' if intervention_has_cycles else 'NO'}", end="")
                
                # Check if we induced repetition
                if intervention_has_cycles and not baseline_has_cycles:
                    text_result['threshold_head_count'] = num_heads
                    text_result['threshold_found'] = True
                    print(f" ✅ THRESHOLD FOUND!")
                    break
                elif intervention_has_cycles and baseline_has_cycles:
                    print(f" (baseline already cyclic)")
                else:
                    print(f" (no cycles)")
                
                # Early stopping if we've tested many heads without success
                if num_heads >= 8 and not intervention_has_cycles:
                    print(f"      → Stopping early after {num_heads} heads (no cycles detected)")
                    break
            
            results['threshold_results'].append(text_result)
            
            # Summary for this text
            if text_result['threshold_found']:
                print(f"   🎯 THRESHOLD: {text_result['threshold_head_count']} heads needed for repetition")
            else:
                print(f"   ❌ NO THRESHOLD: No repetition induced with up to {max_heads} heads")
        
        return results
    
    def create_threshold_analysis_plots(self, results, output_dir):
        """Create visualization plots for threshold analysis."""
        
        # Extract threshold data
        thresholds = []
        threshold_found_count = 0
        
        for text_result in results['threshold_results']:
            if text_result['threshold_found']:
                thresholds.append(text_result['threshold_head_count'])
                threshold_found_count += 1
        
        total_texts = len(results['threshold_results'])
        success_rate = threshold_found_count / total_texts if total_texts > 0 else 0
        
        # Create plots
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Threshold distribution
        plt.subplot(2, 3, 1)
        if thresholds:
            plt.hist(thresholds, bins=max(1, len(set(thresholds))), alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Number of Heads Required')
            plt.ylabel('Frequency')
            plt.title(f'Repetition Threshold Distribution\n(Found in {threshold_found_count}/{total_texts} texts)')
            plt.xticks(range(1, max(thresholds) + 1))
        else:
            plt.text(0.5, 0.5, 'No thresholds found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Repetition Thresholds Found')
        
        # Plot 2: Success rate
        plt.subplot(2, 3, 2)
        categories = ['Threshold Found', 'No Threshold']
        values = [threshold_found_count, total_texts - threshold_found_count]
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
        plt.title(f'Threshold Detection Success\n({success_rate:.1%} success rate)')
        
        # Plot 3: Progression for first few texts
        plt.subplot(2, 3, 3)
        for i, text_result in enumerate(results['threshold_results'][:3]):
            if text_result['progression_results']:
                head_counts = [r['num_heads'] for r in text_result['progression_results']]
                has_cycles = [1 if r['has_cycles'] else 0 for r in text_result['progression_results']]
                
                plt.plot(head_counts, has_cycles, 'o-', label=f'Text {i+1}', alpha=0.7)
        
        plt.xlabel('Number of Heads')
        plt.ylabel('Cycles Detected (0/1)')
        plt.title('Progression Examples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Threshold vs baseline cycles
        plt.subplot(2, 3, 4)
        baseline_has_cycles = [1 if cycles is not None else 0 for cycles in results['baseline_cycles']]
        text_thresholds = []
        
        for text_result in results['threshold_results']:
            if text_result['threshold_found']:
                text_thresholds.append(text_result['threshold_head_count'])
            else:
                text_thresholds.append(0)  # No threshold found
        
        if len(baseline_has_cycles) == len(text_thresholds):
            plt.scatter(baseline_has_cycles, text_thresholds, alpha=0.7, s=100)
            plt.xlabel('Baseline Has Cycles')
            plt.ylabel('Threshold (0 = not found)')
            plt.title('Baseline vs Threshold')
            plt.xticks([0, 1], ['No Cycles', 'Has Cycles'])
        
        # Plot 5: Average threshold by layer
        plt.subplot(2, 3, 5)
        if thresholds:
            avg_threshold = np.mean(thresholds)
            std_threshold = np.std(thresholds)
            
            plt.bar([f'Layer {results["target_layer"]}'], [avg_threshold], 
                   yerr=[std_threshold], capsize=5, alpha=0.7, color='orange')
            plt.ylabel('Average Threshold (heads)')
            plt.title(f'Layer {results["target_layer"]} Performance')
            
            # Add text annotation
            plt.text(0, avg_threshold + std_threshold + 0.1, 
                    f'{avg_threshold:.1f} ± {std_threshold:.1f}', 
                    ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'No thresholds\nfound', ha='center', va='center', 
                    transform=plt.gca().transAxes)
            plt.title(f'Layer {results["target_layer"]}: No Success')
        
        # Plot 6: Detailed progression heatmap
        plt.subplot(2, 3, 6)
        if results['threshold_results']:
            # Create heatmap data
            max_heads_tested = max(len(tr['progression_results']) for tr in results['threshold_results'])
            heatmap_data = np.zeros((len(results['threshold_results']), max_heads_tested))
            
            for i, text_result in enumerate(results['threshold_results']):
                for j, step_result in enumerate(text_result['progression_results']):
                    heatmap_data[i, j] = 1 if step_result['has_cycles'] else 0
            
            sns.heatmap(heatmap_data, cmap='RdYlGn', cbar_kws={'label': 'Cycles Detected'})
            plt.xlabel('Progressive Head Count')
            plt.ylabel('Text Index')
            plt.title('Progression Heatmap')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"threshold_analysis_L{results['target_layer']}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Adaptive Multi-Head Newline Focus Threshold Experiment")
    parser.add_argument("--layer", type=int, default=19, help="Target layer for intervention")
    parser.add_argument("--max_heads", type=int, default=None, help="Maximum heads to test (default: all)")
    parser.add_argument("--strength", type=float, default=2.0, help="Intervention strength")
    parser.add_argument("--focus_multiplier", type=float, default=3.0, help="Newline focus multiplier")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of test samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Adaptive Multi-Head Newline Focus Threshold Experiment")
    print(f"📋 Parameters:")
    print(f"   - Target layer: {args.layer}")
    print(f"   - Max heads: {args.max_heads or 'all'}")
    print(f"   - Intervention strength: {args.strength}")
    print(f"   - Focus multiplier: {args.focus_multiplier}")
    print(f"   - Number of samples: {args.n_samples}")
    
    # Load model and tokenizer
    print(f"🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=args.checkpoint)
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"📚 Loading dataset...")
    try:
        dataset = load_text_dataset("JeanKaddour/minipile")
        test_texts = dataset[:args.n_samples]
    except:
        print("Failed to load JeanKaddour/minipile dataset, using wikitext...")
        dataset = load_text_dataset("wikitext")
        test_texts = dataset[:args.n_samples]
    
    # Filter texts that contain newlines
    newline_texts = []
    for text in test_texts:
        if '\n' in text and len(text.split('\n')) >= 2:
            newline_texts.append(text)
    
    print(f"📊 Found {len(newline_texts)} texts with newlines out of {len(test_texts)} total")
    
    if len(newline_texts) < 5:
        print("⚠️ Adding synthetic texts with newlines for testing...")
        synthetic_texts = [
            "The weather is nice today.\nI think I'll go for a walk.\nMaybe I'll see some birds.",
            "Python is a programming language.\nIt's used for data science.\nMany people love it.",
            "Cats are interesting animals.\nThey sleep a lot during the day.\nAt night they become active.",
            "Books contain knowledge.\nReading expands your mind.\nLibraries are full of books.",
            "Music has many genres.\nRock, jazz, and classical are popular.\nPeople enjoy different styles."
        ]
        newline_texts.extend(synthetic_texts[:max(0, 15 - len(newline_texts))])
    
    test_texts = newline_texts[:args.n_samples]
    print(f"📊 Testing with {len(test_texts)} texts containing newlines")
    
    # Create output directory
    output_dir = Path(f"./plots/adaptive_newline_threshold_L{args.layer}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize processor
    processor = AdaptiveNewlineFocusProcessor(model, tokenizer)
    
    # Run threshold finding experiment
    print(f"🧪 Running adaptive threshold experiment...")
    results = processor.find_repetition_threshold(
        test_texts, args.layer, args.max_heads, 
        intervention_strength=args.strength,
        focus_multiplier=args.focus_multiplier
    )
    
    # Save raw results
    results_path = output_dir / f"adaptive_threshold_results_L{args.layer}.pt"
    torch.save(results, results_path)
    print(f"   ✅ Raw results saved: {results_path}")
    
    # Create analysis plots
    plot_path = processor.create_threshold_analysis_plots(results, output_dir)
    print(f"   ✅ Analysis plots saved: {plot_path}")
    
    # Analyze results
    thresholds = []
    threshold_found_count = 0
    
    for text_result in results['threshold_results']:
        if text_result['threshold_found']:
            thresholds.append(text_result['threshold_head_count'])
            threshold_found_count += 1
    
    total_texts = len(results['threshold_results'])
    success_rate = threshold_found_count / total_texts if total_texts > 0 else 0
    
    # Create detailed report
    report_path = output_dir / f"adaptive_threshold_report_L{args.layer}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Adaptive Multi-Head Newline Focus Threshold Report\n\n")
        f.write(f"**Target Layer**: {args.layer}  \n")
        f.write(f"**Intervention Strength**: {args.strength}  \n")
        f.write(f"**Focus Multiplier**: {args.focus_multiplier}  \n")
        f.write(f"**Total Texts Tested**: {total_texts}  \n\n")
        
        f.write(f"## Threshold Detection Results\n\n")
        f.write(f"- **Success Rate**: {success_rate:.1%} ({threshold_found_count}/{total_texts})\n")
        
        if thresholds:
            avg_threshold = np.mean(thresholds)
            min_threshold = min(thresholds)
            max_threshold = max(thresholds)
            
            f.write(f"- **Average Threshold**: {avg_threshold:.1f} heads\n")
            f.write(f"- **Minimum Threshold**: {min_threshold} heads\n")
            f.write(f"- **Maximum Threshold**: {max_threshold} heads\n")
            f.write(f"- **Threshold Range**: {min_threshold}-{max_threshold} heads\n\n")
            
            if success_rate >= 0.3:
                f.write(f"✅ **STRONG THRESHOLD EFFECT**: {success_rate:.1%} of texts show clear repetition thresholds.\n\n")
                f.write(f"**Key Finding**: Repetition can be reliably induced by forcing {avg_threshold:.1f} heads on average to focus on newlines.\n\n")
            elif success_rate >= 0.1:
                f.write(f"⚠️ **MODERATE THRESHOLD EFFECT**: {success_rate:.1%} of texts show repetition thresholds.\n\n") 
                f.write(f"**Key Finding**: Some texts show thresholds around {avg_threshold:.1f} heads, but effect is inconsistent.\n\n")
            else:
                f.write(f"❌ **WEAK THRESHOLD EFFECT**: Only {success_rate:.1%} of texts show repetition thresholds.\n\n")
        else:
            f.write(f"- **No thresholds detected** across all tested texts\n\n")
            f.write(f"❌ **NO THRESHOLD EFFECT**: Newline focus intervention ineffective for repetition induction.\n\n")
        
        f.write(f"## Mechanism Insights\n\n")
        
        if thresholds:
            f.write(f"- **Threshold Consistency**: {'High' if len(set(thresholds)) <= 3 else 'Moderate' if len(set(thresholds)) <= 6 else 'Low'}\n")
            f.write(f"- **Multi-Head Requirement**: {'Yes' if avg_threshold > 1.5 else 'No'} (avg: {avg_threshold:.1f})\n")
            f.write(f"- **Coordination Effect**: {'Strong' if avg_threshold >= 4 else 'Moderate' if avg_threshold >= 2 else 'Weak'}\n")
        else:
            f.write(f"- **Newline attention insufficient for repetition induction**\n")
            f.write(f"- **Alternative mechanisms likely required**\n")
        
        f.write(f"\n## Individual Text Results\n\n")
        
        for i, text_result in enumerate(results['threshold_results']):
            f.write(f"### Text {i+1}\n")
            f.write(f"- **Input**: {text_result['input_text'][:100]}...\n")
            f.write(f"- **Baseline Cycles**: {'YES' if text_result['baseline_cycles'] is not None else 'NO'}\n")
            if text_result['threshold_found']:
                f.write(f"- **Threshold**: {text_result['threshold_head_count']} heads ✅\n")
            else:
                f.write(f"- **Threshold**: Not found ❌\n")
            f.write(f"- **Max Heads Tested**: {len(text_result['progression_results'])}\n\n")
    
    print(f"   ✅ Report saved: {report_path}")
    
    # Print summary
    print(f"\n🎯 Adaptive Threshold Summary:")
    print(f"   - Success rate: {success_rate:.1%} ({threshold_found_count}/{total_texts})")
    
    if thresholds:
        avg_threshold = np.mean(thresholds)
        print(f"   - Average threshold: {avg_threshold:.1f} heads")
        print(f"   - Threshold range: {min(thresholds)}-{max(thresholds)} heads")
        
        if success_rate >= 0.3:
            print(f"   ✅ STRONG EFFECT: Clear thresholds found!")
        elif success_rate >= 0.1:
            print(f"   ⚠️ MODERATE EFFECT: Some thresholds found")
        else:
            print(f"   ❌ WEAK EFFECT: Few thresholds found")
    else:
        print(f"   ❌ NO THRESHOLDS: Newline focus insufficient")
    
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()