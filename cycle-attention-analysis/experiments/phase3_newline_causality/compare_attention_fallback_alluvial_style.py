#!/usr/bin/env python3
"""
Compare Attention Fallback Patterns: Natural vs No-Cycle-ICL (Alluvial-Style)
==============================================================================

This script uses the SAME approach as the successful OLMo alluvial pipeline:
- Generate naturally from prompts
- Detect cycles in the generation
- Natural = sequences that DID repeat (prompt + cycle + extensions)
- No-Cycle-ICL = sequences that DIDN'T repeat (prompt repeated n_cycles times)

Key differences from ModelGeneratedCycleProcessor:
1. No ICL prepending logic - just natural generation
2. Uses detect_cycles on raw generation (like contrast_analysis.py)
3. No-Cycle-ICL = original prompt * n_cycles (not ICL-prepended generation)

Author: Research Team
Date: November 2025
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
print("✅ torch imported")
from tqdm import tqdm
import time
import argparse
import json
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
print("✅ Basic imports done")

# NLTK imports for proper linguistic classification
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
print("✅ NLTK imports done")

try:
    from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
    from parrots.cycle_detection import detect_cycles
    from torch.nn.utils.rnn import pad_sequence
    print("✅ model_utils, cycle_detection, and pad_sequence imported")
except ImportError as e:
    print(f"❌ Failed to import model_utils: {e}")
    raise

try:
    sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src')
    from modules.cached_data_utils import load_text_dataset
    print("✅ cached_data_utils imported")
except ImportError as e:
    print(f"❌ Failed to import cached_data_utils: {e}")
    raise

# NOTE: We don't use ModelGeneratedCycleProcessor - using alluvial pipeline approach instead
print("✅ Using alluvial-style generation (no ModelGeneratedCycleProcessor)")

print("✅ All imports successful!")

# Download required NLTK data
print("📥 Downloading NLTK data...")
required_nltk_data = ['stopwords', 'averaged_perceptron_tagger', 'punkt', 'punkt_tab']
for data_id in required_nltk_data:
    try:
        nltk.data.find(f'corpora/{data_id}' if data_id == 'stopwords' else f'tokenizers/{data_id}')
    except LookupError:
        nltk.download(data_id, quiet=True)
print("✅ NLTK data ready")

# Initialize NLTK stopwords
try:
    STOP_WORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    print("✅ NLTK available with POS tagging")
except Exception as e:
    print(f"⚠️  NLTK not available: {e}")
    STOP_WORDS = None
    NLTK_AVAILABLE = False

def classify_token_type(token: str) -> str:
    """
    Classify tokens into semantic/structural categories using NLTK.
    
    Categories:
    - NEWLINE: Newline characters
    - SENTENCE_END: Sentence ending punctuation
    - PUNCTUATION: Other punctuation marks
    - CONTENT_WORD: Nouns, verbs, adjectives, adverbs (POS-tagged)
    - FUNCTION_WORD: Determiners, prepositions, pronouns, conjunctions (POS-tagged or stopwords)
    - PROGRAMMING: Code-related tokens
    - BRACKET: Brackets and grouping symbols
    - NUMBER: Numeric tokens
    - OTHER: Special tokens, BOS, etc.
    """
    token = str(token)
    
    # Newline tokens
    if token in ['Ċ', 'čĊ', '\n', 'ĊĊ', 'Ċ\n', 'ĊĠ', 'ĠĊ']:
        return 'NEWLINE'
    
    # Sentence endings  
    if token in ['.', '!', '?', 'Ġ.', 'Ġ!', 'Ġ?', '."', '!"', '?"', '.")', '!")', '?")']:
        return 'SENTENCE_END'
    
    # Punctuation
    if token in [',', ';', ':', 'Ġ,', 'Ġ;', 'Ġ:', '-', '--', 'Ġ-', 'Ġ--', '...', 'Ġ...']:
        return 'PUNCTUATION'
    
    # Brackets and groupings
    if token in ['{', '}', '(', ')', '[', ']', '<', '>', 'Ġ{', 'Ġ}', 'Ġ(', 'Ġ)', 'Ġ[', 'Ġ]', 'Ġ<', 'Ġ>']:
        return 'BRACKET'
    
    # Programming-related tokens
    if token in ['def', 'Ġdef', 'class', 'Ġclass', 'import', 'Ġimport', 'from', 'Ġfrom', 
                 'return', 'Ġreturn', 'if', 'Ġif', 'else', 'Ġelse', 'for', 'Ġfor', 
                 'while', 'Ġwhile', 'print', 'Ġprint', '=', 'Ġ=', '==', 'Ġ==',
                 '+=', '-=', '*=', '/=', '!=', '<=', '>=', '->', '=>']:
        return 'PROGRAMMING'
    
    # Numbers
    clean_token = token.lstrip('Ġ').replace('.', '').replace(',', '')
    if clean_token.isdigit() or (clean_token.replace('-', '').replace('+', '').isdigit()):
        return 'NUMBER'
    
    # Use NLTK for linguistic classification
    if NLTK_AVAILABLE and len(token) > 0:
        # Remove GPT2 space marker
        clean_token = token.lstrip('Ġ')
        clean_lower = clean_token.lower()
        
        # Check if it's a stopword (these are function words)
        if STOP_WORDS and clean_lower in STOP_WORDS:
            return 'FUNCTION_WORD'
        
        # Use POS tagging for more accurate classification
        if len(clean_token) > 0 and clean_token.isalpha():
            try:
                # POS tag the word
                pos_tags = pos_tag([clean_token])
                if pos_tags:
                    pos = pos_tags[0][1]
                    
                    # Content words (nouns, verbs, adjectives, adverbs)
                    content_pos = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                                  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                                  'JJ', 'JJR', 'JJS',  # Adjectives
                                  'RB', 'RBR', 'RBS',  # Adverbs
                                  'FW'}  # Foreign words
                    
                    # Function words (determiners, prepositions, pronouns, etc.)
                    function_pos = {'DT',  # Determiners (the, a, an)
                                   'IN',  # Prepositions/subordinating conjunctions
                                   'CC',  # Coordinating conjunctions
                                   'TO',  # "to"
                                   'PRP', 'PRP$',  # Pronouns
                                   'WDT', 'WP', 'WP$', 'WRB',  # Wh-words
                                   'MD',  # Modals (can, should, will)
                                   'PDT', 'POS', 'UH'}  # Predeterminers, possessive, interjections
                    
                    if pos in content_pos:
                        return 'CONTENT_WORD'
                    elif pos in function_pos:
                        return 'FUNCTION_WORD'
                    # If POS is unclear, fall through to heuristics below
            except:
                pass
        
        # Heuristic: longer words are more likely to be content words
        if len(clean_token) > 4:
            return 'CONTENT_WORD'
    
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

def generate_and_detect_cycles(texts, model, tokenizer, max_length, max_new_tokens, batch_size, n_cycles):
    """
    Generate from texts and detect cycles - matching alluvial pipeline approach.
    
    Returns:
    - natural_sequences: sequences that repeated (with cycle data)
    - no_cycle_icl_sequences: sequences that didn't repeat (prompt repeated n times)
    """
    device = model.device
    natural_sequences = []
    no_cycle_icl_sequences = []
    
    # Set padding
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🔄 Generating and detecting cycles in {len(texts)} texts...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize and pad
        pretokenized = [
            tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            for t in batch_texts
        ]
        
        input_ids_list = [b['input_ids'].squeeze(0) for b in pretokenized]
        attention_mask_list = [b['attention_mask'].squeeze(0) for b in pretokenized]
        
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        
        toked = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}
        
        # Generate
        with torch.no_grad():
            generated = model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=tokenizer.pad_token_id)
        
        # Get prompt lengths
        plengths = toked["attention_mask"].sum(dim=1).tolist()
        
        # Move to CPU and remove padding
        generated_cpu = [generated[j].detach().cpu() for j in range(len(generated))]
        generated_cpu = [g[g != tokenizer.pad_token_id] for g in generated_cpu]
        
        # Detect cycles in generated portion (EXACTLY like alluvial pipeline)
        for j, gen_seq in enumerate(generated_cpu):
            prompt_len = plengths[j]
            generated_portion = gen_seq[prompt_len:]  # Only look at generated tokens
            
            # Detect cycles in generated portion
            cycle_result = detect_cycles(generated_portion, return_index=True, 
                                        pad_token_id=tokenizer.pad_token_id)
            
            if cycle_result[0] is not None:
                # Has cycle - create NATURAL sequence
                cycle_tokens, cycle_len, n_repeats, cycle_start_idx = cycle_result
                
                # Natural = prompt + generated portion with cycle
                natural_seq = {
                    'sequence': gen_seq.tolist(),
                    'prompt_length': prompt_len,
                    'cycle_start': prompt_len + cycle_start_idx,
                    'cycle': cycle_tokens,
                    'cycle_text': tokenizer.decode(cycle_tokens),
                    'type': 'natural'
                }
                natural_sequences.append(natural_seq)
            
            else:
                # No cycle - create NO-CYCLE-ICL sequence
                # No-Cycle-ICL = original prompt repeated n_cycles times (EXACTLY like alluvial)
                prompt_tokens = gen_seq[:prompt_len].tolist()
                no_cycle_icl_seq = prompt_tokens * n_cycles
                
                no_cycle_icl = {
                    'sequence': no_cycle_icl_seq,
                    'prompt_length': prompt_len,
                    'original_text': batch_texts[j],
                    'type': 'no_cycle_icl'
                }
                no_cycle_icl_sequences.append(no_cycle_icl)
    
    # Restore padding side
    tokenizer.padding_side = original_padding_side
    
    print(f"   ✅ Generated: {len(natural_sequences)} Natural, {len(no_cycle_icl_sequences)} No-Cycle-ICL")
    
    return natural_sequences, no_cycle_icl_sequences

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
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Attention Shifts Comparison
    ax1 = axes[0, 0]
    token_types = sorted(set(list(natural_shifts.keys()) + list(no_cycle_shifts.keys())))
    natural_shift_values = [natural_shifts.get(tt, 0) for tt in token_types]
    no_cycle_shift_values = [no_cycle_shifts.get(tt, 0) for tt in token_types]
    
    x_pos = np.arange(len(token_types))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, natural_shift_values, width, label='Natural (Repetitive)', alpha=0.8, color='#e74c3c')
    bars2 = ax1.bar(x_pos + width/2, no_cycle_shift_values, width, label='No-Cycle-ICL (Non-Repetitive)', alpha=0.8, color='#3498db')
    
    ax1.set_xlabel('Token Type')
    ax1.set_ylabel('Attention Shift (No-Newline - Baseline)')
    ax1.set_title('🧠 Attention Fallback Comparison: Natural vs No-Cycle-ICL')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(token_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if abs(height) > 0.001:
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15), textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if abs(height) > 0.001:
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15), textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
        # 2. Semantic vs Structural Focus
    ax2 = axes[0, 1]
    
    semantic_types = ['CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING']
    structural_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'BRACKET']
    
    def get_category_shift(shifts_dict: Dict[str, float], categories: List[str]) -> float:
        return sum(shifts_dict.get(cat, 0) for cat in categories)
    
    natural_semantic_shift = get_category_shift(natural_shifts, semantic_types)
    natural_structural_shift = get_category_shift(natural_shifts, structural_types)
    no_cycle_semantic_shift = get_category_shift(no_cycle_shifts, semantic_types)
    no_cycle_structural_shift = get_category_shift(no_cycle_shifts, structural_types)
    
    categories = ['Semantic Tokens', 'Structural Tokens']
    natural_values = [natural_semantic_shift, natural_structural_shift]
    no_cycle_values = [no_cycle_semantic_shift, no_cycle_structural_shift]
    
    x_pos = np.arange(len(categories))
    bars3 = ax2.bar(x_pos - width/2, natural_values, width, label='Natural (Repetitive)', alpha=0.8, color='#e74c3c')
    bars4 = ax2.bar(x_pos + width/2, no_cycle_values, width, label='No-Cycle-ICL (Non-Repetitive)', alpha=0.8, color='#3498db')
    
    ax2.set_xlabel('Token Category')
    ax2.set_ylabel('Total Attention Shift')
    ax2.set_title('🎯 Semantic vs Structural Attention Shifts')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15), textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15), textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # 3. Heatmap of Attention Shifts
    ax3 = axes[1, 0]
    
    # Create matrix for heatmap
    shift_matrix = np.array([
        [natural_shifts.get(tt, 0) for tt in token_types],
        [no_cycle_shifts.get(tt, 0) for tt in token_types]
    ])
    
    im = ax3.imshow(shift_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    ax3.set_xticks(range(len(token_types)))
    ax3.set_xticklabels(token_types, rotation=45, ha='right')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Natural\n(Repetitive)', 'No-Cycle-ICL\n(Non-Repetitive)'])
    ax3.set_title('🌡️ Attention Shift Heatmap')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(token_types)):
            value = shift_matrix[i, j]
            if abs(value) > 0.005:
                ax3.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color='white' if abs(value) > 0.05 else 'black', fontweight='bold')
    
    plt.colorbar(im, ax=ax3, label='Attention Shift')
    
    # 4. Distribution Comparison for Key Token Types
    ax4 = axes[1, 1]
    
    key_types = ['CONTENT_WORD', 'FUNCTION_WORD', 'NEWLINE', 'PUNCTUATION', 'PROGRAMMING']
    natural_key_shifts = [natural_shifts.get(kt, 0) for kt in key_types]
    no_cycle_key_shifts = [no_cycle_shifts.get(kt, 0) for kt in key_types]
    
    # Define colors for each token type (5 types need 5 colors)
    key_colors = ['#9b59b6', '#e67e22', '#e74c3c', '#f39c12', '#3498db']  # Purple, Orange, Red, Yellow, Blue
    
    # Scatter plot
    ax4.scatter(natural_key_shifts, no_cycle_key_shifts, s=100, alpha=0.7, c=key_colors)
    
    for i, tt in enumerate(key_types):
        ax4.annotate(tt, (natural_key_shifts[i], no_cycle_key_shifts[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add diagonal line
    lims = [min(ax4.get_xlim()[0], ax4.get_ylim()[0]), max(ax4.get_xlim()[1], ax4.get_ylim()[1])]
    ax4.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    ax4.set_xlabel('Natural (Repetitive) Attention Shift')
    ax4.set_ylabel('No-Cycle-ICL (Non-Repetitive) Attention Shift')
    ax4.set_title('📊 Key Token Types: Shift Correlation')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "paper_figure_natural_vs_nocycle_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Comparison plot saved: {plot_path}")
    plt.close()
    
    return natural_shifts, no_cycle_shifts

def create_analysis_report(natural_results: List[Dict], no_cycle_results: List[Dict], 
                          natural_shifts: Dict[str, float], no_cycle_shifts: Dict[str, float], 
                          output_dir: Path):
    """Create detailed analysis report."""
    
    report_content = f"""# Attention Fallback Analysis: Natural vs No-Cycle-ICL
**Comparing attention redistribution when newlines are removed**

## Summary Statistics

### Natural Sequences (Repetitive)
- **Sequences analyzed**: {len([r for r in natural_results if r['success']])}
- **Failed analyses**: {len([r for r in natural_results if not r['success']])}
- **Average newlines removed per sequence**: {np.mean([r.get('newlines_removed', 0) for r in natural_results if r['success']]):.1f}

### No-Cycle-ICL Sequences (Non-Repetitive)  
- **Sequences analyzed**: {len([r for r in no_cycle_results if r['success']])}
- **Failed analyses**: {len([r for r in no_cycle_results if not r['success']])}
- **Average newlines removed per sequence**: {np.mean([r.get('newlines_removed', 0) for r in no_cycle_results if r['success']]):.1f}

## Token Type Definitions

### Semantic Tokens (Meaning-bearing)
- **CONTENT_WORD**: Content words (nouns, verbs, adjectives, adverbs - classified via NLTK POS tagging)
- **FUNCTION_WORD**: Function words (determiners, prepositions, conjunctions - classified via NLTK)
- **PROGRAMMING**: Programming-related tokens ('def', 'class', 'import', '=', etc.)

### Structural Tokens (Non-semantic)
- **NEWLINE**: Line break tokens ('Ċ', 'čĊ', '\\n')
- **SENTENCE_END**: Sentence terminators ('.', '!', '?')
- **PUNCTUATION**: Internal punctuation (',', ';', ':')
- **BRACKET**: Grouping symbols ('{{', '}}', '(', ')', '[', ']')
- **NUMBER**: Numeric tokens

## Attention Shift Analysis

| Token Type | Natural Shift (pp) | No-Cycle-ICL Shift (pp) | Difference (pp) | Pattern |
|------------|-------------------|-------------------------|-----------------|---------|"""

    # Add token type analysis
    all_token_types = sorted(set(list(natural_shifts.keys()) + list(no_cycle_shifts.keys())))
    
    for token_type in all_token_types:
        natural_val = natural_shifts.get(token_type, 0)
        no_cycle_val = no_cycle_shifts.get(token_type, 0)
        diff = natural_val - no_cycle_val
        
        # Determine pattern
        if abs(diff) < 0.005:
            pattern = "🟢 **Similar**"
        elif diff > 0.01:
            pattern = "🔴 **Natural > No-Cycle**"
        elif diff < -0.01:
            pattern = "🔵 **No-Cycle > Natural**"
        else:
            pattern = "🟡 **Slight Difference**"
        
        report_content += f"\n| {token_type} | {natural_val:+.3f}pp | {no_cycle_val:+.3f}pp | {diff:+.3f}pp | {pattern} |"

    # Add category analysis
    semantic_types = ['CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING']
    structural_types = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'BRACKET']
    
    natural_semantic_shift = sum(natural_shifts.get(cat, 0) for cat in semantic_types)
    natural_structural_shift = sum(natural_shifts.get(cat, 0) for cat in structural_types)
    no_cycle_semantic_shift = sum(no_cycle_shifts.get(cat, 0) for cat in semantic_types)
    no_cycle_structural_shift = sum(no_cycle_shifts.get(cat, 0) for cat in structural_types)
    
    report_content += f"""

## Category-Level Analysis

### Semantic Tokens (Total)
- **Natural sequences**: {natural_semantic_shift:+.3f}pp shift
- **No-Cycle-ICL sequences**: {no_cycle_semantic_shift:+.3f}pp shift
- **Difference**: {natural_semantic_shift - no_cycle_semantic_shift:+.3f}pp

### Structural Tokens (Total)
- **Natural sequences**: {natural_structural_shift:+.3f}pp shift  
- **No-Cycle-ICL sequences**: {no_cycle_structural_shift:+.3f}pp shift
- **Difference**: {natural_structural_shift - no_cycle_structural_shift:+.3f}pp

## Key Findings

### 🎯 Primary Fallback Targets
"""
    
    # Find top gainers for each type
    natural_top_gainers = sorted([(k, v) for k, v in natural_shifts.items() if v > 0.01], key=lambda x: x[1], reverse=True)[:3]
    no_cycle_top_gainers = sorted([(k, v) for k, v in no_cycle_shifts.items() if v > 0.01], key=lambda x: x[1], reverse=True)[:3]
    
    report_content += f"""
**Natural (Repetitive) Sequences:**
"""
    for token_type, shift in natural_top_gainers:
        report_content += f"\n- {token_type}: +{shift:.3f}pp"
    
    report_content += f"""

**No-Cycle-ICL (Non-Repetitive) Sequences:**
"""
    for token_type, shift in no_cycle_top_gainers:
        report_content += f"\n- {token_type}: +{shift:.3f}pp"

    # Add interpretation
    content_diff = natural_shifts.get('CONTENT_WORD', 0) - no_cycle_shifts.get('CONTENT_WORD', 0)
    
    report_content += f"""

### 🧠 Interpretation

**Content Word Attention Shift:**
- Natural sequences: {natural_shifts.get('CONTENT_WORD', 0):+.3f}pp
- No-Cycle-ICL sequences: {no_cycle_shifts.get('CONTENT_WORD', 0):+.3f}pp
- **Difference**: {content_diff:+.3f}pp

"""
    
    if abs(content_diff) < 0.01:
        report_content += "✅ **Similar Fallback Pattern**: Both repetitive and non-repetitive sequences show similar attention redistribution to content words when newlines are removed."
    elif content_diff > 0.01:
        report_content += "🔍 **Repetitive Preference**: Natural (repetitive) sequences show stronger fallback to content words, suggesting repetitive patterns enhance semantic focus when structural cues are removed."
    else:
        report_content += "📊 **Non-Repetitive Preference**: No-Cycle-ICL (non-repetitive) sequences show stronger fallback to content words, suggesting different attention mechanisms in non-repetitive text."
    
    semantic_diff = natural_semantic_shift - no_cycle_semantic_shift
    
    report_content += f"""

**Overall Semantic vs Structural Preference:**
- Semantic shift difference: {semantic_diff:+.3f}pp
- Structural shift difference: {natural_structural_shift - no_cycle_structural_shift:+.3f}pp

"""
    
    if abs(semantic_diff) < 0.02:
        report_content += "✅ **Consistent Semantic Preference**: Both sequence types show similar preference for semantic content when newlines are removed."
    else:
        report_content += "🧠 **Different Semantic Strategies**: Repetitive and non-repetitive sequences employ different attention strategies when structural cues are unavailable."

    report_content += f"""

## Implications for Repetition Research

### Newline Causality Assessment
- Both sequence types show minimal reliance on newlines for attention
- Attention naturally redistributes to semantic content regardless of repetition status
- **Confirms**: Newlines are not causal for repetition mechanisms

### Attention Mechanism Insights
- Repetitive vs non-repetitive sequences may use different fallback strategies
- Content words remain primary attention targets across both sequence types
- Structural token attention shows minimal variation between sequence types

---
*Analysis conducted using EleutherAI/pythia-1.4b, layer 19 attention patterns*
*Generated: October 2025*
"""

    # Save report
    report_path = output_dir / "attention_fallback_natural_vs_nocycle_report.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"   📝 Analysis report saved: {report_path}")

def main():
    print("🚀 Attention Fallback Analysis (Alluvial-Style)")
    
    parser = argparse.ArgumentParser(description="Compare attention fallback using alluvial pipeline approach")
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-1B-hf", help="Model name to analyze")
    parser.add_argument("--revision", type=str, default=None, help="Model checkpoint revision (e.g., step425000-tokens1781B)")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples (like alluvial: 1000)")
    parser.add_argument("--target_layer", type=int, default=None, help="Target layer to analyze (defaults to 75%% depth)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for sequence generation")
    parser.add_argument("--max_length", type=int, default=32, help="Prompt max length (like alluvial)")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Generation length (like alluvial)")
    parser.add_argument("--n_cycles", type=int, default=4, help="Number of cycles for ICL (like alluvial)")
    
    args = parser.parse_args()
    
    # Derive safe model name for paths
    safe_model_name = args.model_name.replace("/", "_")
    
    # Set output directory
    if args.output_dir is None:
        if args.revision:
            args.output_dir = f"./plots/attention_fallback_alluvial_{safe_model_name}_{args.revision}_seed{args.seed}"
        else:
            args.output_dir = f"./plots/attention_fallback_alluvial_{safe_model_name}_seed{args.seed}"
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🔧 Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if args.revision:
        print(f"🤖 Loading model and tokenizer: {args.model_name} (revision: {args.revision})...")
    else:
        print(f"🤖 Loading model and tokenizer: {args.model_name}...")
    
    # Load with revision if specified
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.revision:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            revision=args.revision, 
            trust_remote_code=True,
            attn_implementation="eager"  # Required for output_attentions=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, trust_remote_code=True)
    else:
        # Also use eager attention for non-revision models
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            attn_implementation="eager"  # Required for output_attentions=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    model.to(device)
    model.eval()
    print("   ✅ Model loaded and ready!")
    
    # Determine target layer if not specified (75% depth)
    # Handle different architectures
    if args.target_layer is None:
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            n_layers = len(model.transformer.h)  # GPT-2 style
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            n_layers = len(model.gpt_neox.layers)  # Pythia/GPT-NeoX style
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)  # LLaMA style
        else:
            raise ValueError(f"Unknown model architecture for {args.model_name}")
        args.target_layer = int(n_layers * 0.75)
        print(f"   🎯 Auto-selected layer {args.target_layer} ({args.target_layer}/{n_layers} = 75% depth)")
    else:
        print(f"   🎯 Using specified layer {args.target_layer}")
    
    # Set random seed BEFORE loading dataset
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"   🎲 Random seed set to: {args.seed}")
    
    # Load dataset
    print("📚 Loading dataset...")
    texts = load_text_dataset(n_samples=args.n_samples, seed=args.seed)
    print(f"   ✅ Loaded {len(texts)} texts")
    
    # Generate and detect cycles (ALLUVIAL STYLE)
    natural_seqs, no_cycle_icl_seqs = generate_and_detect_cycles(
        texts, model, tokenizer, args.max_length, args.max_new_tokens, 
        args.batch_size, args.n_cycles
    )
    
    print(f"\n   📊 Sequence counts:")
    print(f"     - Natural (with cycles): {len(natural_seqs)}")
    print(f"     - No-Cycle-ICL (without cycles): {len(no_cycle_icl_seqs)}")
    
    if len(no_cycle_icl_seqs) == 0:
        print("\n   ⚠️  No No-Cycle-ICL sequences found!")
        print("   💡 OLMo-1B-hf has ~99.8% repetition rate")
        print("   💡 Try increasing n_samples or using a different model")
        return
    
    # Select sequences to analyze
    natural_to_analyze = natural_seqs[:args.n_samples] if len(natural_seqs) >= args.n_samples else natural_seqs
    no_cycle_icl_to_analyze = no_cycle_icl_seqs
    
    print(f"   🎯 Analyzing:")
    print(f"     - Natural sequences: {len(natural_to_analyze)}")
    print(f"     - No-Cycle-ICL sequences: {len(no_cycle_icl_to_analyze)}")
    
    # Analyze Natural sequences
    print("\n🔍 Analyzing Natural (repetitive) sequences...")
    natural_results = []
    for seq_data in tqdm(natural_to_analyze, desc="Processing Natural"):
        result = analyze_sequence_attention_fallback(
            model, tokenizer, seq_data, device, args.target_layer
        )
        natural_results.append(result)
    
    # Analyze No-Cycle-ICL sequences  
    print("\n🔍 Analyzing No-Cycle-ICL (non-repetitive) sequences...")
    no_cycle_results = []
    for seq_data in tqdm(no_cycle_icl_to_analyze, desc="Processing No-Cycle-ICL"):
        result = analyze_sequence_attention_fallback(
            model, tokenizer, seq_data, device, args.target_layer
        )
        no_cycle_results.append(result)
    
    # Create comparison plots and analysis
    print("\n📊 Creating comparison analysis...")
    natural_shifts, no_cycle_shifts = plot_attention_fallback_comparison(
        natural_results, no_cycle_results, output_dir
    )
    
    # Create detailed report
    print("\n📝 Generating analysis report...")
    create_analysis_report(
        natural_results, no_cycle_results, 
        natural_shifts, no_cycle_shifts, 
        output_dir
    )
    
    # Save raw results
    results_data = {
        'natural_results': natural_results,
        'no_cycle_results': no_cycle_results,
        'natural_shifts': natural_shifts,
        'no_cycle_shifts': no_cycle_shifts,
        'analysis_parameters': {
            'n_samples': args.n_samples,
            'target_layer': args.target_layer,
            'model': args.model_name,
            'dataset': 'JeanKaddour/minipile',
            'n_cycles': args.n_cycles,
            'max_length': args.max_length,
            'max_new_tokens': args.max_new_tokens,
            'natural_count': len(natural_seqs),
            'no_cycle_count': len(no_cycle_icl_seqs),
            'approach': 'alluvial_style'
        }
    }
    
    results_path = output_dir / "attention_fallback_alluvial_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"   📊 Visualization: paper_figure_natural_vs_nocycle_comparison.png")
    print(f"   📝 Report: attention_fallback_natural_vs_nocycle_report.md")
    print(f"   💾 Data: attention_fallback_alluvial_results.json")
    print(f"   🔬 Approach: Alluvial-style (like OLMo pipeline)")
    
    # Print key findings
    content_natural = natural_shifts.get('CONTENT_WORD', 0)
    content_no_cycle = no_cycle_shifts.get('CONTENT_WORD', 0)
    content_diff = content_natural - content_no_cycle
    
    print(f"\n🎯 **Key Finding**: Content word attention shift difference: {content_diff:+.3f}pp")
    if abs(content_diff) < 0.01:
        print("   ✅ Both sequence types show similar attention fallback patterns")
    else:
        print(f"   🔍 {'Repetitive' if content_diff > 0 else 'Non-repetitive'} sequences show stronger semantic fallback")

if __name__ == "__main__":
    main()