#!/usr/bin/env python3
"""
Generate a summary report of the important heads analysis.
"""

print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
from pathlib import Path
import re
import argparse

def generate_summary_report(contrast_path, cycle_data_path, output_path, top_k=10):
    """Generate a summary report of the important heads analysis."""
    
    print(f"📊 Generating summary report...")
    
    # Extract contrast values
    contrast_data = {}
    
    for layer_dir in sorted(Path(contrast_path).glob("layer_*")):
        layer_num = int(layer_dir.name.split('_')[1])
        log_files = list(layer_dir.glob(f"*_cyc3_*32.out"))
        if not log_files:
            continue
            
        with open(log_files[0], 'r') as f:
            log_content = f.read()
        
        natural_match = re.search(fr"layer {layer_num} natural heatmap: (.+)", log_content)
        no_cycle_icl_match = re.search(fr"layer {layer_num} no-cycle icl heatmap: (.+)", log_content)
        
        if natural_match:
            natural_values = np.array(eval(natural_match.group(1)))
            contrast_data.setdefault(layer_num, {})['natural'] = natural_values
            
        if no_cycle_icl_match:
            no_cycle_icl_values = np.array(eval(no_cycle_icl_match.group(1)))
            contrast_data.setdefault(layer_num, {})['no_cycle_icl'] = no_cycle_icl_values
    
    # Identify important heads
    all_contrasts = []
    for layer_num, layer_data in contrast_data.items():
        for seq_type, values in layer_data.items():
            if seq_type in ['natural', 'no_cycle_icl']:
                for head_idx, contrast in enumerate(values):
                    all_contrasts.append((abs(contrast), layer_num, head_idx, seq_type, contrast))
    
    all_contrasts.sort(reverse=True)
    
    # Get top K unique (layer, head) pairs
    important_heads = []
    seen_heads = set()
    for contrast_val, layer_num, head_idx, seq_type, raw_contrast in all_contrasts:
        head_pair = (layer_num, head_idx)
        if head_pair not in seen_heads and len(important_heads) < top_k:
            important_heads.append((layer_num, head_idx, contrast_val, seq_type, raw_contrast))
            seen_heads.add(head_pair)
    
    # Generate report
    report = []
    report.append("# Important Attention Heads Analysis Report")
    report.append("## Summary")
    report.append(f"- Analysis based on contrast values at cycle size 3")
    report.append(f"- Template size: 32")
    report.append(f"- Top {top_k} heads identified for focused analysis")
    report.append(f"- Sequence types considered: natural, no-cycle ICL")
    report.append("")
    
    report.append("## Top Important Heads")
    report.append("| Rank | Layer | Head | Contrast Value | Sequence Type | Raw Contrast |")
    report.append("|------|-------|------|----------------|---------------|--------------|")
    
    for rank, (layer, head, contrast_val, seq_type, raw_contrast) in enumerate(important_heads, 1):
        report.append(f"| {rank:2d} | {layer:2d} | {head:2d} | {contrast_val:.2e} | {seq_type:12s} | {raw_contrast:+.2e} |")
    
    report.append("")
    report.append("## Analysis Details")
    report.append(f"- **Contrast Path**: {contrast_path}")
    report.append(f"- **Cycle Data Path**: {cycle_data_path}")
    report.append(f"- **Total Layers Processed**: {len(contrast_data)}")
    report.append(f"- **Heads per Layer**: 16")
    report.append("")
    
    report.append("## Layer Distribution")
    layer_counts = {}
    for layer, head, _, _, _ in important_heads:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    for layer in sorted(layer_counts.keys()):
        count = layer_counts[layer]
        report.append(f"- **Layer {layer}**: {count} head{'s' if count > 1 else ''}")
    
    report.append("")
    report.append("## Generated Files")
    report.append("- `natural_focus_evolution_important_heads.png`: Focus token evolution for natural sequences")
    report.append("- `natural_attention_evolution_important_heads.png`: Attention distribution evolution for natural sequences")  
    report.append("- `no_cycle_icl_focus_evolution_important_heads.png`: Focus token evolution for no-cycle ICL sequences")
    report.append("- `no_cycle_icl_attention_evolution_important_heads.png`: Attention distribution evolution for no-cycle ICL sequences")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Report generated: {output_path}")
    
    return important_heads

def main():
    parser = argparse.ArgumentParser(description="Generate summary report for important heads")
    parser.add_argument("--contrast_path", type=str, 
                       default="/home/mmahaut/projects/parrots/outputs_multihead_full/EleutherAI/pythia-1.4b/steplatest",
                       help="Path to contrast data logs")
    parser.add_argument("--cycle_data_path", type=str,
                       default="/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/cycle_evolution_parametric/cycles_4/steplatest",
                       help="Path to cycle evolution data")
    parser.add_argument("--output_path", type=str, 
                       default="./plots/focused_important_heads/analysis_report.md",
                       help="Output path for the report")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top heads to analyze")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    
    generate_summary_report(
        args.contrast_path,
        args.cycle_data_path, 
        args.output_path,
        args.top_k
    )

if __name__ == "__main__":
    main()