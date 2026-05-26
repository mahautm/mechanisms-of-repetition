#!/usr/bin/env python3
"""
Analyze prompt token composition vs attention patterns using pre-computed cycle evolution data.
This script loads existing .pt files and compares prompt composition with attention focus patterns.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import argparse

# NLP libraries for proper word classification
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data
    required_nltk_data = ['stopwords', 'averaged_perceptron_tagger', 'punkt']
    for data_name in required_nltk_data:
        try:
            nltk.data.find(f'corpora/{data_name}' if data_name in ['stopwords', 'punkt'] else f'taggers/{data_name}')
        except LookupError:
            print(f"📥 Downloading NLTK {data_name}...")
            nltk.download(data_name, quiet=True)
    
    STOP_WORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    print("✅ NLTK available with POS tagging")
except ImportError:
    print("⚠️  NLTK not available, using basic classification")
    STOP_WORDS = set()
    NLTK_AVAILABLE = False

print("✅ All imports successful!")

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

def classify_token_type(token):
    """Classify tokens into meaningful categories using NLTK where possible."""
    # Strip Ġ prefix for easier matching (GPT-2 style space prefix)
    clean_token = token.replace('Ġ', '').replace(' ', '')
    clean_lower = clean_token.lower()
    
    # NEWLINE - various newline representations
    if token in ['Ċ', 'čĊ', 'ĊĊ', '\n', '\\n']:
        return 'NEWLINE'
    
    # SENTENCE_END - sentence-ending punctuation
    if clean_token in ['.', '!', '?', '...', '?!', '!?']:
        return 'SENTENCE_END'
    
    # PUNCTUATION - commas, semicolons, colons, dashes, quotes
    if clean_token in [',', ';', ':', '-', '—', '–', '"', "'", '"', '"', ''', ''', '`', 
                       '--', '---', ')', '(', ']', '[', '}', '{']:
        return 'PUNCTUATION'
    
    # BRACKET - standalone brackets
    if clean_token in ['{', '}', '(', ')', '[', ']', '<', '>']:
        return 'BRACKET'
    
    # PROGRAMMING - code-specific tokens (using sets for O(1) lookup)
    programming_keywords = {
        'function', 'def', 'class', 'import', 'from', 'return', 'export', 'module',
        'if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'break', 'continue',
        'const', 'let', 'var', 'public', 'private', 'protected', 'static',
        'void', 'int', 'float', 'double', 'string', 'bool', 'char', 'long',
        'true', 'false', 'null', 'none', 'undefined', 'nil',
        'async', 'await', 'try', 'catch', 'finally', 'throw', 'except',
        'self', 'this', 'super', 'new', 'delete', 'typeof', 'instanceof',
        'print', 'println', 'printf', 'console', 'log', 'debug',
    }
    
    programming_operators = {
        '=', '==', '!=', '===', '!==', '+=', '-=', '*=', '/=', '%=',
        '&&', '||', '&', '|', '^', '~', '<<', '>>', 
        '=>', '->', '::', '..', 
        '<?', '?>', '<%', '%>', '{%', '%}', '{{', '}}',
        '#', '//', '/*', '*/', '<!--', '-->',
    }
    
    if clean_lower in programming_keywords or clean_token in programming_operators:
        return 'PROGRAMMING'
    
    # Tab characters and special formatting
    if token in ['ĉ', 'čĊĉ', 'Ċĉ']:
        return 'PROGRAMMING'
    
    # NUMBER - numeric values
    if clean_token.replace('.', '').replace('-', '').replace('+', '').replace(',', '').isdigit():
        return 'NUMBER'
    
    # For alphabetic tokens, use NLTK for proper classification
    if clean_token.isalpha() or (token.startswith('Ġ') and clean_token.isalpha()):
        # First check NLTK stopwords (function words)
        if STOP_WORDS and clean_lower in STOP_WORDS:
            return 'FUNCTION_WORD'  # Stopwords are function words
        
        # Use NLTK POS tagging if available to identify content words
        if NLTK_AVAILABLE and len(clean_token) > 1:
            try:
                # POS tag the word
                pos_tags = pos_tag([clean_token])
                pos = pos_tags[0][1] if pos_tags else None
                
                # Content word POS tags (nouns, verbs, adjectives, adverbs)
                content_pos = {
                    'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                    'JJ', 'JJR', 'JJS',  # Adjectives
                    'RB', 'RBR', 'RBS',  # Adverbs
                    'FW',  # Foreign words
                }
                
                # Function word POS tags
                function_pos = {
                    'DT',  # Determiners (the, a, an)
                    'IN',  # Prepositions/subordinating conjunctions (in, of, like)
                    'CC',  # Coordinating conjunctions (and, but, or)
                    'TO',  # to
                    'PRP', 'PRP$',  # Pronouns (he, she, it, his, her)
                    'WDT', 'WP', 'WP$', 'WRB',  # Wh-words (who, what, where, when)
                    'MD',  # Modal verbs (can, could, will, would)
                    'PDT',  # Predeterminers (all, both)
                    'POS',  # Possessive ending ('s)
                    'UH',  # Interjections (uh, oh)
                }
                
                if pos in content_pos:
                    return 'CONTENT_WORD'
                elif pos in function_pos:
                    return 'FUNCTION_WORD'  # Function words get their own category
                # If POS is unclear, fall through to heuristics below
                    
            except Exception:
                pass  # Fall through to heuristics
        
        # Fallback heuristics if NLTK not available or POS tagging fails
        # Longer words are more likely to be content words
        if len(clean_token) > 1:
            return 'CONTENT_WORD'
    
    # Tokens with mixed alphanumeric (e.g., "covid19", "mp3")
    if clean_token and any(c.isalpha() for c in clean_token) and any(c.isdigit() for c in clean_token):
        if len(clean_token) > 2:
            return 'CONTENT_WORD'
        else:
            return 'OTHER'
    
    # Everything else (BOS, EOS, special tokens, symbols, etc.)
    return 'OTHER'

def load_tokenizer():
    """Load tokenizer for token analysis."""
    try:
        from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer
        print("📦 Loading tokenizer...")
        model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
        print("✅ Tokenizer loaded!")
        del model  # Free up memory since we only need tokenizer
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        raise

def analyze_prompt_composition_from_data(cycle_data, tokenizer):
    """Extract prompt token composition from cycle evolution data."""
    print("📊 Analyzing prompt token composition from cycle data...")
    
    prompt_stats = defaultdict(list)
    
    for seq_type, seq_results in cycle_data.items():
        print(f"  🔍 Processing {seq_type} sequences...")
        
        sequences = seq_results.get('sequences', [])
        
        for seq_data in sequences:
            # Get prompt tokens
            prompt_length = seq_data.get('prompt_length', len(seq_data['sequence']) // 4)
            prompt_token_ids = seq_data['sequence'][:prompt_length]
            
            try:
                prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_token_ids)
            except Exception as e:
                print(f"     ⚠️  Error converting tokens: {e}")
                continue
            
            # Count token types in this prompt
            token_counts = defaultdict(int)
            total_tokens = len(prompt_tokens)
            
            for token in prompt_tokens:
                token_type = classify_token_type(token)
                token_counts[token_type] += 1
            
            # Calculate percentages for this prompt
            for token_type in ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING', 'BRACKET', 'NUMBER', 'OTHER']:
                percentage = (token_counts[token_type] / total_tokens * 100) if total_tokens > 0 else 0
                prompt_stats[f"{seq_type}_{token_type}"].append(percentage)
    
    return prompt_stats

def analyze_attention_focus_from_data(cycle_data, tokenizer):
    """Extract attention focus patterns from cycle evolution data."""
    print("🔍 Analyzing attention focus patterns from cycle data...")
    
    attention_stats = defaultdict(list)
    
    for seq_type, seq_results in cycle_data.items():
        print(f"  🎯 Processing {seq_type} attention patterns...")
        
        focus_tokens_data = seq_results.get('focus_tokens', [])
        sequences = seq_results.get('sequences', [])
        
        # Aggregate focus patterns across all sequences and heads
        all_focused_tokens = []
        
        for seq_idx, seq_focus_tokens in enumerate(focus_tokens_data):
            if seq_idx >= len(sequences):
                continue
                
            seq_data = sequences[seq_idx]
            
            # Get all tokens in the sequence for context
            try:
                all_tokens = tokenizer.convert_ids_to_tokens(seq_data['sequence'])
            except Exception as e:
                print(f"     ⚠️  Error converting sequence tokens: {e}")
                continue
            
            # Collect focused tokens from all heads
            for head_focus_tokens in seq_focus_tokens:
                for token_info in head_focus_tokens:
                    token = token_info.get('token', '')
                    attention_weight = token_info.get('attention_weight', 0)
                    
                    # Weight the token by attention strength
                    all_focused_tokens.append((token, attention_weight))
        
        # Count weighted token types
        token_type_weights = defaultdict(float)
        total_weight = 0
        
        for token, weight in all_focused_tokens:
            token_type = classify_token_type(token)
            token_type_weights[token_type] += weight
            total_weight += weight
        
        # Convert to percentages
        for token_type in ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING', 'BRACKET', 'NUMBER', 'OTHER']:
            percentage = (token_type_weights[token_type] / total_weight * 100) if total_weight > 0 else 0
            attention_stats[f"{seq_type}_{token_type}"].append(percentage)
    
    return attention_stats

def create_prompt_vs_attention_comparison_plot(prompt_stats, attention_stats, output_dir, layer_num, cycles_num):
    """Create comprehensive comparison plots for all sequence types."""
    
    categories = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING', 'BRACKET', 'NUMBER', 'OTHER']
    
    # Determine which sequence types have data
    available_seq_types = []
    for seq_type in ['natural', 'no_cycle_icl']:
        # Check if this sequence type has any data
        has_data = any(f"{seq_type}_{cat}" in prompt_stats for cat in categories)
        if has_data:
            available_seq_types.append(seq_type)
    
    if not available_seq_types:
        print("   ⚠️  No data available for plotting")
        return None
    
    # Colors for consistency
    colors = {
        'NEWLINE': '#e74c3c',
        'SENTENCE_END': '#2ecc71',
        'PUNCTUATION': '#f39c12',
        'CONTENT_WORD': '#9b59b6',
        'FUNCTION_WORD': '#e67e22',  # Orange for function words
        'PROGRAMMING': '#3498db',
        'BRACKET': '#34495e',
        'NUMBER': '#f1c40f',
        'OTHER': '#95a5a6'
    }
    
    # Create figure with subplots for each sequence type (dynamic rows)
    n_rows = len(available_seq_types)
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
    
    # Ensure axes is 2D even with single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for seq_idx, seq_type in enumerate(available_seq_types):
        # Calculate prompt composition statistics
        prompt_means = []
        prompt_stds = []
        for category in categories:
            key = f"{seq_type}_{category}"
            values = prompt_stats.get(key, [0])
            prompt_means.append(np.mean(values) if values else 0)
            prompt_stds.append(np.std(values) if len(values) > 1 else 0)
        
        # Calculate attention statistics
        attention_means = []
        attention_stds = []
        for category in categories:
            key = f"{seq_type}_{category}"
            values = attention_stats.get(key, [0])
            attention_means.append(np.mean(values) if values else 0)
            attention_stds.append(np.std(values) if len(values) > 1 else 0)
        
        bar_colors = [colors.get(cat, '#95a5a6') for cat in categories]
        
        # Plot 1: Prompt composition
        ax1 = axes[seq_idx, 0]
        bars1 = ax1.bar(range(len(categories)), prompt_means, yerr=prompt_stds,
                        color=bar_colors, alpha=0.7, capsize=5)
        ax1.set_title(f'Prompt Token Composition\n{seq_type.replace("_", " ").title()} Sequences', 
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(prompt_means, prompt_stds)):
            ax1.text(i, mean + std + 0.5, f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 2: Attention focus
        ax2 = axes[seq_idx, 1]
        bars2 = ax2.bar(range(len(categories)), attention_means, yerr=attention_stds,
                        color=bar_colors, alpha=0.7, capsize=5)
        ax2.set_title(f'Attention Focus Distribution\nLayer {layer_num} - {seq_type.replace("_", " ").title()}', 
                      fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(attention_means, attention_stds)):
            ax2.text(i, mean + std + 0.5, f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 3: Ratio comparison (attention/prompt)
        ax3 = axes[seq_idx, 2]
        ratios = []
        for prompt_mean, attention_mean in zip(prompt_means, attention_means):
            if prompt_mean > 0:
                ratio = attention_mean / prompt_mean
            else:
                ratio = 0 if attention_mean == 0 else float('inf')
            ratios.append(ratio)
        
        # Cap infinite ratios for visualization
        ratios_capped = [min(r, 10) if r != float('inf') else 10 for r in ratios]
        
        bars3 = ax3.bar(range(len(categories)), ratios_capped, color=bar_colors, alpha=0.7)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Proportional (1:1)')
        ax3.set_title(f'Attention Bias Ratio\n(Attention % / Prompt %)', 
                      fontsize=14, fontweight='bold')
        ax3.set_ylabel('Ratio (>1 = Over-attention)', fontsize=12)
        ax3.set_xticks(range(len(categories)))
        ax3.set_xticklabels(categories, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()
        
        # Add ratio labels
        for i, (ratio, ratio_capped) in enumerate(zip(ratios, ratios_capped)):
            if ratio == float('inf'):
                label = '∞'
            elif ratio > 10:
                label = f'>{ratio:.1f}'
            else:
                label = f'{ratio:.2f}'
            ax3.text(i, ratio_capped + 0.1, label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # plt.suptitle(f'Prompt Composition vs Attention Focus Analysis\nLayer {layer_num}, {cycles_num} Cycles', 
    #              fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save plot
    comparison_path = output_dir / f"prompt_vs_attention_comparison_L{layer_num}_C{cycles_num}.png"
    
    try:
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Comparison plot saved: {comparison_path}")
        
        if comparison_path.exists():
            file_size = comparison_path.stat().st_size
            print(f"   📁 File size: {file_size} bytes")
        
    except Exception as e:
        print(f"   ❌ Error saving comparison plot: {e}")
        fallback_path = Path("/tmp") / comparison_path.name
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        print(f"   🚨 Saved to fallback: {fallback_path}")
    finally:
        plt.close()
    
    return comparison_path

def create_summary_statistics_report(prompt_stats, attention_stats, output_dir, layer_num, cycles_num):
    """Create a text report with key statistics."""
    
    categories = ['NEWLINE', 'SENTENCE_END', 'PUNCTUATION', 'CONTENT_WORD', 'FUNCTION_WORD', 'PROGRAMMING', 'BRACKET', 'NUMBER', 'OTHER']
    seq_types = ['natural', 'no_cycle_icl']
    
    report = []
    report.append(f"# Prompt vs Attention Analysis Report")
    report.append(f"**Layer {layer_num}, {cycles_num} Cycles**")
    report.append(f"")
    
    for seq_type in seq_types:
        report.append(f"## {seq_type.replace('_', ' ').title()} Sequences")
        report.append(f"")
        
        # Calculate statistics
        prompt_data = {}
        attention_data = {}
        
        for category in categories:
            key = f"{seq_type}_{category}"
            prompt_values = prompt_stats.get(key, [0])
            attention_values = attention_stats.get(key, [0])
            
            prompt_data[category] = {
                'mean': np.mean(prompt_values) if prompt_values else 0,
                'std': np.std(prompt_values) if len(prompt_values) > 1 else 0,
                'n': len(prompt_values)
            }
            
            attention_data[category] = {
                'mean': np.mean(attention_values) if attention_values else 0,
                'std': np.std(attention_values) if len(attention_values) > 1 else 0,
                'n': len(attention_values)
            }
        
        # Create table
        report.append("| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |")
        report.append("|------------|----------|-------------|------------|----------------|")
        
        for category in categories:
            prompt_mean = prompt_data[category]['mean']
            attention_mean = attention_data[category]['mean']
            
            if prompt_mean > 0:
                bias_ratio = attention_mean / prompt_mean
            else:
                bias_ratio = float('inf') if attention_mean > 0 else 0
            
            # Determine interpretation
            if bias_ratio > 2:
                interpretation = "🔴 **Over-attended**"
            elif bias_ratio > 1.2:
                interpretation = "🟡 **Moderately over-attended**"
            elif bias_ratio < 0.5:
                interpretation = "🔵 **Under-attended**"
            elif bias_ratio < 0.8:
                interpretation = "🟡 **Moderately under-attended**"
            else:
                interpretation = "🟢 **Proportional**"
            
            bias_str = f"{bias_ratio:.2f}" if bias_ratio != float('inf') else "∞"
            
            report.append(f"| {category} | {prompt_mean:.1f}% | {attention_mean:.1f}% | {bias_str} | {interpretation} |")
        
        report.append("")
        
        # Key findings
        sorted_by_bias = sorted(categories, key=lambda c: attention_data[c]['mean'] / max(prompt_data[c]['mean'], 0.001), reverse=True)
        
        report.append("### Key Findings:")
        report.append(f"- **Most over-attended**: {sorted_by_bias[0]}")
        report.append(f"- **Most under-attended**: {sorted_by_bias[-1]}")
        
        # Find specializations
        high_attention = [c for c in categories if attention_data[c]['mean'] > 20]
        if high_attention:
            report.append(f"- **High attention categories** (>20%): {', '.join(high_attention)}")
        
        report.append("")
    
    # Save report
    report_path = output_dir / f"prompt_vs_attention_report_L{layer_num}_C{cycles_num}.md"
    
    try:
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"   ✅ Summary report saved: {report_path}")
    except Exception as e:
        print(f"   ❌ Error saving report: {e}")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Analyze prompt composition vs attention patterns")
    parser.add_argument("--data_path", type=str, 
                       default="/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/cycle_evolution_parametric/cycles_4/steplatest",
                       help="Path to cycle evolution data directory")
    parser.add_argument("--output_dir", type=str, default="./plots/prompt_vs_attention_analysis",
                       help="Output directory for plots and reports")
    parser.add_argument("--layer", type=int, default=None, 
                       help="Layer number to analyze (extracted from filename if not provided)")
    parser.add_argument("--cycles", type=int, default=None,
                       help="Cycles number (extracted from path if not provided)")
    
    args = parser.parse_args()
    
    # Set up paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Starting prompt vs attention analysis...")
    print(f"📂 Data path: {data_path}")
    print(f"📁 Output path: {output_dir}")
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Find and load cycle evolution data files
    pt_files = list(data_path.glob("cycle_evolution_parametric_c*_l*_*.pt"))
    
    if not pt_files:
        print(f"❌ No cycle evolution data files found in {data_path}")
        print(f"   Looking for files matching: cycle_evolution_parametric_c*_l*_*.pt")
        return
    
    print(f"📊 Found {len(pt_files)} data files")
    
    for pt_file in pt_files:
        print(f"\n🔍 Processing: {pt_file.name}")
        
        # Extract layer and cycles from filename
        import re
        match = re.search(r'_c(\d+)_l(\d+)_', pt_file.name)
        if not match:
            print(f"   ⚠️  Could not extract layer/cycles from filename")
            continue
            
        cycles_num = int(match.group(1))
        layer_num = int(match.group(2))
        
        # Override with command line args if provided
        if args.layer is not None:
            layer_num = args.layer
        if args.cycles is not None:
            cycles_num = args.cycles
        
        print(f"   📊 Analyzing Layer {layer_num}, {cycles_num} cycles")
        
        try:
            # Load cycle evolution data
            cycle_data = torch.load(pt_file, map_location='cpu')
            print(f"   ✅ Loaded data with sequence types: {list(cycle_data.keys())}")
            
            # Analyze prompt composition
            print(f"   📊 Analyzing prompt composition...")
            prompt_stats = analyze_prompt_composition_from_data(cycle_data, tokenizer)
            
            # Analyze attention focus
            print(f"   🎯 Analyzing attention focus...")
            attention_stats = analyze_attention_focus_from_data(cycle_data, tokenizer)
            
            # Create comparison plots
            print(f"   📈 Creating comparison plots...")
            plot_path = create_prompt_vs_attention_comparison_plot(
                prompt_stats, attention_stats, output_dir, layer_num, cycles_num
            )
            
            # Create summary report
            print(f"   📝 Creating summary report...")
            report_path = create_summary_statistics_report(
                prompt_stats, attention_stats, output_dir, layer_num, cycles_num
            )
            
            print(f"   ✅ Completed analysis for Layer {layer_num}")
            
        except Exception as e:
            print(f"   ❌ Error processing {pt_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Prompt vs attention analysis complete!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()