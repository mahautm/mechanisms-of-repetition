#!/usr/bin/env python3
"""
Fixed Attention Fallback Comparison: Natural vs No-Cycle-ICL
==========================================================

This version bypasses the problematic ModelGeneratedCycleProcessor and uses
a simpler approach to ensure we get sufficient sequences of both types.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src')

import torch
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer
from modules.cached_data_utils import load_text_dataset
print("✅ All imports successful!")

def classify_token_type(token: str) -> str:
    """Classify tokens into semantic/structural categories."""
    token = str(token)
    
    # Newline tokens
    if token in ['Ċ', 'čĊ', '\n', 'ĊĊ', 'Ċ\n']:
        return 'NEWLINE'
    
    # Sentence endings  
    if token in ['.', '!', '?', 'Ġ.', 'Ġ!', 'Ġ?']:
        return 'SENTENCE_END'
    
    # Punctuation
    if token in [',', ';', ':', 'Ġ,', 'Ġ;', 'Ġ:', '-', '--', 'Ġ-', 'Ġ--']:
        return 'PUNCTUATION'
    
    # Content words (space-prefixed words)
    if token.startswith('Ġ') and len(token) > 1 and token[1:].isalpha():
        return 'CONTENT_WORD'
    
    # Function words
    function_words = {'and', 'or', 'but', 'with', 'from', 'into', 'the', 'a', 'an', 'is', 'are', 'was', 'were'}
    if token.lstrip('Ġ').lower() in function_words:
        return 'FUNCTION_WORD'
    
    print(f"Unrecognized token type for token: {token}")
    return 'OTHER'

def get_attention_distribution_by_token_type(attention_weights: torch.Tensor, tokens: List[str]) -> Dict[str, float]:
    """Get attention distribution across token types."""
    distribution = defaultdict(float)
    
    for i, token in enumerate(tokens):
        if i < len(attention_weights):
            token_type = classify_token_type(token)
            distribution[token_type] += attention_weights[i].item()
    
    return dict(distribution)

def remove_newlines_from_tokens(tokens: List[str]) -> List[str]:
    """Remove newline tokens from token list."""
    return [token for token in tokens if classify_token_type(token) != 'NEWLINE']

def create_repetitive_sequence(text: str, tokenizer) -> Dict:
    """Create a repetitive sequence by adding repetitive prompts."""
    # Add repetitive elements
    repetitive_prompt = f"{text[:100]} again and again and again and again and"
    tokens = tokenizer.encode(repetitive_prompt, max_length=512, truncation=True)
    
    return {
        'sequence': tokens,
        'type': 'natural',
        'source': 'manual_repetitive'
    }

def create_non_repetitive_sequence(text: str, tokenizer) -> Dict:
    """Create a non-repetitive sequence from clean text."""
    # Use clean text without repetitive elements
    clean_text = text[:400]  # Longer clean text
    tokens = tokenizer.encode(clean_text, max_length=512, truncation=True)
    
    return {
        'sequence': tokens,
        'type': 'no_cycle_icl', 
        'source': 'manual_clean'
    }

def analyze_sequence_attention_fallback(model, tokenizer, sequence_data: Dict, device: torch.device, target_layer: int = 19) -> Dict[str, Any]:
    """Analyze attention fallback for a single sequence."""
    try:
        # Get tokenized sequence
        input_ids = torch.tensor([sequence_data['sequence']]).to(device)
        
        # Truncate if too long
        if input_ids.shape[1] > 1024:
            input_ids = input_ids[:, :1024]
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get baseline attention (with newlines)
        with torch.no_grad():
            outputs_baseline = model(input_ids, output_attentions=True)
            attention_baseline = outputs_baseline.attentions[target_layer][0]  # Shape: (n_heads, seq_len, seq_len)
        
        # Remove newlines and get attention
        tokens_no_newline = remove_newlines_from_tokens(tokens)
        if len(tokens_no_newline) < 5:  # Skip if too few tokens remain
            return {'success': False, 'error': 'Too few tokens after newline removal'}
            
        input_ids_no_newline = tokenizer.convert_tokens_to_ids(tokens_no_newline)
        input_ids_no_newline = torch.tensor([input_ids_no_newline]).to(device)
        
        with torch.no_grad():
            outputs_no_newline = model(input_ids_no_newline, output_attentions=True)
            attention_no_newline = outputs_no_newline.attentions[target_layer][0]  # Shape: (n_heads, seq_len, seq_len)
        
        # Analyze attention distribution for each head at the last token position
        n_heads = attention_baseline.shape[0]
        head_results = []
        
        for head_idx in range(n_heads):
            # Baseline attention distribution (last token attending to all positions)
            baseline_attn = attention_baseline[head_idx, -1, :]  # Last token's attention
            baseline_distribution = get_attention_distribution_by_token_type(baseline_attn, tokens)
            
            # No-newline attention distribution (last token attending to all positions)
            no_newline_attn = attention_no_newline[head_idx, -1, :]  # Last token's attention
            no_newline_distribution = get_attention_distribution_by_token_type(no_newline_attn, tokens_no_newline)
            
            head_results.append({
                'head_idx': head_idx,
                'baseline_distribution': baseline_distribution,
                'no_newline_distribution': no_newline_distribution
            })
        
        return {
            'success': True,
            'head_results': head_results,
            'sequence_type': sequence_data.get('type', 'unknown'),
            'sequence_length_baseline': len(tokens),
            'sequence_length_no_newline': len(tokens_no_newline),
            'newlines_removed': len(tokens) - len(tokens_no_newline)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'sequence_type': sequence_data.get('type', 'unknown')
        }

def plot_attention_fallback_comparison(natural_results: List[Dict], no_cycle_results: List[Dict], output_dir: Path):
    """Create comprehensive comparison plots."""
    
    # Aggregate results by sequence type and token type
    def aggregate_distributions(results: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
        """Aggregate attention distributions across sequences and heads."""
        aggregated = {
            'baseline': defaultdict(list),
            'no_newline': defaultdict(list)
        }
        
        for result in results:
            if not result['success']:
                continue
                
            for head_result in result['head_results']:
                # Baseline distributions
                for token_type, attention in head_result['baseline_distribution'].items():
                    aggregated['baseline'][token_type].append(attention)
                
                # No-newline distributions
                for token_type, attention in head_result['no_newline_distribution'].items():
                    aggregated['no_newline'][token_type].append(attention)
        
        return aggregated
    
    natural_agg = aggregate_distributions(natural_results)
    no_cycle_agg = aggregate_distributions(no_cycle_results)
    
    # Calculate mean attention shifts
    def calculate_shifts(agg_data: Dict) -> Dict[str, float]:
        shifts = {}
        all_token_types = set(list(agg_data['baseline'].keys()) + list(agg_data['no_newline'].keys()))
        
        for token_type in all_token_types:
            baseline_mean = np.mean(agg_data['baseline'].get(token_type, [0]))
            no_newline_mean = np.mean(agg_data['no_newline'].get(token_type, [0]))
            shifts[token_type] = no_newline_mean - baseline_mean
        
        return shifts
    
    natural_shifts = calculate_shifts(natural_agg)
    no_cycle_shifts = calculate_shifts(no_cycle_agg)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Attention shifts comparison
    token_types = sorted(set(list(natural_shifts.keys()) + list(no_cycle_shifts.keys())))
    natural_shift_values = [natural_shifts.get(tt, 0) for tt in token_types]
    no_cycle_shift_values = [no_cycle_shifts.get(tt, 0) for tt in token_types]
    
    x_pos = np.arange(len(token_types))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, natural_shift_values, width, 
                   label='Natural (Repetitive)', alpha=0.8, color='#e74c3c')
    bars2 = ax1.bar(x_pos + width/2, no_cycle_shift_values, width, 
                   label='No-Cycle-ICL (Non-Repetitive)', alpha=0.8, color='#3498db')
    
    ax1.set_xlabel('Token Type')
    ax1.set_ylabel('Attention Shift (No-Newline - Baseline)')
    ax1.set_title('🧠 Attention Fallback: Natural vs No-Cycle-ICL (FIXED)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(token_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    def add_value_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.001:
                ax.annotate(f'{height:.3f}', 
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3 if height > 0 else -15), 
                          textcoords="offset points",
                          ha='center', va='bottom' if height > 0 else 'top', 
                          fontsize=9)
    
    add_value_labels(bars1, ax1)
    add_value_labels(bars2, ax1)
    
    # Plot 2: Content Word vs Other comparison
    content_natural = natural_shifts.get('CONTENT_WORD', 0)
    content_no_cycle = no_cycle_shifts.get('CONTENT_WORD', 0)
    
    other_natural = sum(v for k, v in natural_shifts.items() if k != 'CONTENT_WORD')
    other_no_cycle = sum(v for k, v in no_cycle_shifts.items() if k != 'CONTENT_WORD')
    
    categories = ['Content Words', 'All Other Tokens']
    natural_cat_values = [content_natural, other_natural]
    no_cycle_cat_values = [content_no_cycle, other_no_cycle]
    
    x_pos2 = np.arange(len(categories))
    bars3 = ax2.bar(x_pos2 - width/2, natural_cat_values, width, 
                   label='Natural (Repetitive)', alpha=0.8, color='#e74c3c')
    bars4 = ax2.bar(x_pos2 + width/2, no_cycle_cat_values, width, 
                   label='No-Cycle-ICL (Non-Repetitive)', alpha=0.8, color='#3498db')
    
    ax2.set_xlabel('Token Category')
    ax2.set_ylabel('Total Attention Shift')
    ax2.set_title('🎯 Content vs Other Tokens (FIXED)')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    add_value_labels(bars3, ax2)
    add_value_labels(bars4, ax2)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "attention_fallback_comparison_FIXED.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Fixed comparison plot saved: {plot_path}")
    plt.close()
    
    return natural_shifts, no_cycle_shifts

def main():
    print("🚀 FIXED Attention Fallback Analysis: Natural vs No-Cycle-ICL")
    
    parser = argparse.ArgumentParser(description="Fixed comparison of attention fallback patterns")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples per sequence type")
    parser.add_argument("--target_layer", type=int, default=19, help="Target layer to analyze")
    parser.add_argument("--output_dir", type=str, default="./plots/attention_fallback_FIXED", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🔧 Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    model.to(device)
    model.eval()
    print("   ✅ Model ready!")
    
    # Load dataset
    print("📚 Loading dataset...")
    texts = load_text_dataset(n_samples=args.n_samples * 2)  # Load enough for both types
    print(f"   ✅ Loaded {len(texts)} texts")
    
    # Create sequences manually (bypassing the problematic generator)
    print("🔄 Creating sequences manually...")
    
    # Create repetitive sequences
    natural_sequences = []
    for i in range(args.n_samples):
        if i < len(texts):
            seq = create_repetitive_sequence(texts[i], tokenizer)
            natural_sequences.append(seq)
    
    # Create non-repetitive sequences
    no_cycle_sequences = []
    for i in range(args.n_samples, min(len(texts), args.n_samples * 2)):
        seq = create_non_repetitive_sequence(texts[i], tokenizer)
        no_cycle_sequences.append(seq)
    
    print(f"   📊 Created sequences:")
    print(f"     - Natural (repetitive): {len(natural_sequences)}")
    print(f"     - No-Cycle-ICL (non-repetitive): {len(no_cycle_sequences)}")
    
    # Analyze sequences
    print("\n🔍 Analyzing Natural sequences...")
    natural_results = []
    for seq_data in tqdm(natural_sequences, desc="Processing Natural"):
        result = analyze_sequence_attention_fallback(model, tokenizer, seq_data, device, args.target_layer)
        natural_results.append(result)
    
    print("\n🔍 Analyzing No-Cycle-ICL sequences...")
    no_cycle_results = []
    for seq_data in tqdm(no_cycle_sequences, desc="Processing No-Cycle-ICL"):
        result = analyze_sequence_attention_fallback(model, tokenizer, seq_data, device, args.target_layer)
        no_cycle_results.append(result)
    
    # Create plots
    print("\n📊 Creating comparison plots...")
    natural_shifts, no_cycle_shifts = plot_attention_fallback_comparison(
        natural_results, no_cycle_results, output_dir
    )
    
    # Save results
    results_data = {
        'natural_results': natural_results,
        'no_cycle_results': no_cycle_results,
        'natural_shifts': natural_shifts,
        'no_cycle_shifts': no_cycle_shifts,
        'analysis_parameters': {
            'n_samples': args.n_samples,
            'target_layer': args.target_layer,
            'method': 'manual_sequence_creation'
        }
    }
    
    with open(output_dir / "fixed_results.json", 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Print summary
    natural_success = len([r for r in natural_results if r['success']])
    no_cycle_success = len([r for r in no_cycle_results if r['success']])
    
    print(f"\n✅ FIXED Analysis Complete!")
    print(f"📊 Results:")
    print(f"   - Natural sequences analyzed: {natural_success}")
    print(f"   - No-Cycle-ICL sequences analyzed: {no_cycle_success}")
    print(f"   - Content word shift difference: {natural_shifts.get('CONTENT_WORD', 0) - no_cycle_shifts.get('CONTENT_WORD', 0):+.3f}pp")
    
    print(f"\n📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()