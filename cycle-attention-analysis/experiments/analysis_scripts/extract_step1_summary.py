#!/usr/bin/env python3
"""
Lightweight script to extract summary statistics from step1 data without loading everything into memory.
This avoids the OOM issues while still getting the key metrics for evolution analysis.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

def classify_token_type(token):
    """Classify tokens into meaningful categories for analysis."""
    if token in ['Ċ', 'čĊ']:
        return 'NEWLINE'
    elif token in ['.', '!', '?']:
        return 'SENTENCE_END'
    elif token in [',', ';', ':']:
        return 'PUNCTUATION'
    elif token in ['The', 'Hello', 'Python', 'Machine', 'Att']:
        return 'TEMPLATE_WORD'
    elif token.startswith('Ġ'):  # GPT-style space prefix
        return 'CONTENT_WORD'
    elif token in ['{', '}', '(', ')', '[', ']']:
        return 'BRACKET'
    elif token.isdigit() or any(c.isdigit() for c in token):
        return 'NUMBER'
    else:
        return 'OTHER'

def extract_key_metrics_from_layer(data_file):
    """Extract key bias metrics from a single layer file to avoid memory issues."""
    try:
        data = torch.load(data_file, map_location='cpu')
        
        # Handle different data formats (step1 vs later steps)
        if 'focus_evolution_data' in data:
            # Later steps format
            focus_data = data['focus_evolution_data']
            token_names = data.get('token_names', [])
        elif 'natural' in data and 'icl' in data:
            # Step1 format - need to process raw attention data
            focus_data = {}
            token_names = []
            
            # Extract tokens from natural scenario
            if 'focus_tokens' in data['natural']:
                # Get unique tokens across all cycles - handle nested structure
                all_tokens = []
                focus_tokens_data = data['natural']['focus_tokens']
                # This is deeply nested: heads -> cycles -> tokens
                for head_data in focus_tokens_data:
                    if isinstance(head_data, list):
                        for cycle_data in head_data:
                            if isinstance(cycle_data, list):
                                for token_entry in cycle_data:
                                    if isinstance(token_entry, dict) and 'token' in token_entry:
                                        all_tokens.append(str(token_entry['token']))
                token_names = list(set(all_tokens)) if all_tokens else []
            
            # For step1, we need to calculate focus ratios from raw attention data
            for scenario in ['natural', 'icl']:
                if scenario in data and 'attention_distributions' in data[scenario]:
                    # Calculate average attention for each token
                    scenario_focus = {}
                    attention_dists = data[scenario]['attention_distributions']
                    focus_tokens_list = data[scenario]['focus_tokens']
                    
                    # Process the complex nested structure of step1 data
                    # focus_tokens_list is a list of attention heads, each containing lists of cycles
                    for head_idx, head_data in enumerate(focus_tokens_list):
                        if isinstance(head_data, list):
                            # Each head contains multiple cycles with token dictionaries
                            for cycle_data in head_data:
                                if isinstance(cycle_data, list):
                                    # Each cycle contains token dictionaries with attention weights
                                    for token_entry in cycle_data:
                                        if isinstance(token_entry, dict) and 'token' in token_entry and 'attention_weight' in token_entry:
                                            token = str(token_entry['token'])
                                            attention_weight = float(token_entry['attention_weight'])
                                            
                                            if token not in scenario_focus:
                                                scenario_focus[token] = []
                                            scenario_focus[token].append(attention_weight)
                    
                    # Average attention per token (this is our "focus ratio")
                    avg_focus = {}
                    for token, values in scenario_focus.items():
                        if values:
                            avg_focus[token] = np.mean(values)
                    
                    focus_data[scenario] = avg_focus
        else:
            return None
        
        # Calculate bias ratios for key token types
        bias_ratios = {}
        
        # Process each scenario (natural, icl)
        for scenario in ['natural', 'icl']:
            if scenario not in focus_data:
                continue
                
            scenario_data = focus_data[scenario]
            ratios = {}
            
            # Handle different data structures
            if isinstance(scenario_data, dict):
                # Step1 format: dict of token -> attention value
                for token, attention_val in scenario_data.items():
                    token_type = classify_token_type(str(token))
                    if token_type not in ratios:
                        ratios[token_type] = []
                    ratios[token_type].append(attention_val)
            elif isinstance(scenario_data, (list, np.ndarray)) and token_names:
                # Later steps format: array indexed by token_names
                for token_idx, token in enumerate(token_names):
                    if token_idx < len(scenario_data):
                        token_type = classify_token_type(token)
                        if token_type not in ratios:
                            ratios[token_type] = []
                        ratios[token_type].append(scenario_data[token_idx])
            
            # Average ratios by type
            avg_ratios = {}
            for token_type, values in ratios.items():
                if values:
                    avg_ratios[token_type] = np.mean(values)
            
            bias_ratios[scenario] = avg_ratios
        
        # Clear memory
        del data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return bias_ratios
        
    except Exception as e:
        import traceback
        print(f"Error processing {data_file}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    data_dir = Path("/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/cycle_evolution_parametric/cycles_4/step1")
    output_dir = Path("./plots/multi_step_analysis/bias_summary_step1")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🔍 Extracting key metrics from step1 data...")
    print(f"📂 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Find all layer files
    layer_files = sorted(list(data_dir.glob("cycle_evolution_parametric_c4_l*_all_results.pt")))
    
    if not layer_files:
        print("❌ No data files found!")
        return
    
    print(f"📊 Found {len(layer_files)} layer files")
    
    # Collect metrics from each layer
    all_metrics = {}
    
    for i, layer_file in enumerate(layer_files):
        print(f"🔄 Processing layer {i+1}/{len(layer_files)}: {layer_file.name}")
        
        layer_num = int(layer_file.name.split('_l')[1].split('_')[0])
        metrics = extract_key_metrics_from_layer(layer_file)
        
        if metrics:
            all_metrics[layer_num] = metrics
        else:
            print(f"   ⚠️ Failed to extract metrics from layer {layer_num}")
    
    # Calculate overall statistics
    print(f"📈 Calculating summary statistics...")
    
    summary_stats = {
        'natural': defaultdict(list),
        'icl': defaultdict(list)
    }
    
    for layer_num, layer_metrics in all_metrics.items():
        for scenario in ['natural', 'icl']:
            if scenario in layer_metrics:
                for token_type, ratio in layer_metrics[scenario].items():
                    summary_stats[scenario][token_type].append(ratio)
    
    # Create final summary
    final_summary = {}
    for scenario in ['natural', 'icl']:
        final_summary[scenario] = {}
        for token_type, values in summary_stats[scenario].items():
            if values:
                final_summary[scenario][token_type] = {
                    'mean': float(np.mean(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values)),
                    'std': float(np.std(values))
                }
    
    # Save the summary
    summary_file = output_dir / "attention_bias_statistical_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# Attention Bias Statistical Summary - Step1\n\n")
        f.write("## Bias Ratio Statistics by Token Type and Scenario\n\n")
        
        for scenario in ['natural', 'icl']:
            f.write(f"### {scenario.upper()} Scenario\n\n")
            f.write("| Token Type | Mean Ratio | Max Ratio | Min Ratio | Std Dev |\n")
            f.write("|------------|------------|-----------|-----------|----------|\n")
            
            if scenario in final_summary:
                for token_type, stats in final_summary[scenario].items():
                    f.write(f"| {token_type} | {stats['mean']:.3f} | {stats['max']:.3f} | {stats['min']:.3f} | {stats['std']:.3f} |\n")
            f.write("\n")
        
        # Add key metrics for evolution analysis
        if 'natural' in final_summary:
            natural_stats = final_summary['natural']
            newline_bias = natural_stats.get('NEWLINE', {}).get('max', 0)
            content_bias = natural_stats.get('CONTENT_WORD', {}).get('mean', 1)
            template_bias = natural_stats.get('TEMPLATE_WORD', {}).get('max', 1)
            
            f.write("## Key Evolution Metrics\n\n")
            f.write(f"- **Newline Bias (Natural)**: {newline_bias:.2f}x\n")
            f.write(f"- **Content Bias (Natural)**: {content_bias:.2f}x\n")
            f.write(f"- **Template Max (Natural)**: {template_bias:.2f}x\n")
            
            if 'icl' in final_summary and 'NEWLINE' in final_summary['icl']:
                icl_newline = final_summary['icl']['NEWLINE'].get('max', 0)
                f.write(f"- **Newline Bias (ICL)**: {icl_newline:.2f}x\n")
    
    # Also save as JSON for programmatic access
    json_file = output_dir / "step1_summary.json"
    with open(json_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"✅ Summary statistics saved to: {summary_file}")
    print(f"✅ JSON data saved to: {json_file}")
    print(f"📊 Processed {len(all_metrics)} layers successfully")

if __name__ == "__main__":
    main()