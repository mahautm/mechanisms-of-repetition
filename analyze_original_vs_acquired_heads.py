#!/usr/bin/env python3
"""
Analyze multihead attention differences between original (innate) vs acquired repetition.

Original: sequences that were already repeating at step1
Acquired: sequences that first started repeating at a later checkpoint

This requires running inference to get per-sample attention head contrasts.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import re
from collections import defaultdict
import argparse

def load_cycle_evolution_data(results_file):
    """Load cycle evolution JSON data"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return {cp: {int(k): v for k, v in status.items()} 
            for cp, status in data.items()}

def classify_samples(all_status, checkpoints):
    """
    Classify each sample as original, acquired, or never-repeating.
    
    Returns dict: {sample_idx: {'category': 'original'|'acquired'|'never', 'first_cp': checkpoint}}
    """
    n_texts = max(max(status.keys()) for status in all_status.values() if status) + 1
    
    classification = {}
    
    for idx in range(n_texts):
        first_rep_cp = None
        for cp in checkpoints:
            if cp in all_status and idx in all_status[cp]:
                if all_status[cp][idx]:  # is repeating
                    first_rep_cp = cp
                    break
        
        if first_rep_cp is None:
            classification[idx] = {'category': 'never', 'first_cp': None}
        elif first_rep_cp == checkpoints[0]:  # step1
            classification[idx] = {'category': 'original', 'first_cp': first_rep_cp}
        else:
            classification[idx] = {'category': 'acquired', 'first_cp': first_rep_cp}
    
    return classification

def parse_multihead_output(output_file, layer):
    """Parse multihead analysis output to extract heatmap and repetition indices."""
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Extract heatmap (8 heads for pythia-70m, 16 for pythia-1.4b)
    heatmap_match = re.search(rf'layer {layer} natural heatmap: \[(.*?)\]', content, re.DOTALL)
    if heatmap_match:
        heatmap_str = heatmap_match.group(1).replace('\n', ' ')
        heatmap = np.array([float(x.strip()) for x in heatmap_str.split(',') if x.strip()])
    else:
        heatmap = None
    
    # Extract repetition index 
    rep_match = re.search(rf'layer {layer} repetition index: \[(.*?)\]', content, re.DOTALL)
    if rep_match:
        rep_str = rep_match.group(1).replace('\n', ' ')
        rep_index = [int(x.strip()) for x in rep_str.split(',') if x.strip()]
    else:
        rep_index = None
    
    return heatmap, rep_index

def run_per_sample_analysis(model_name, checkpoint, layer, classification, device='cuda'):
    """
    Run analysis to get per-sample attention head contrasts.
    
    This runs the model and computes attention head contrasts for each sample,
    then groups them by original vs acquired.
    """
    from parrots.aa_fortu.modules.model_utils import HookedModel, load_model_and_tokenizer
    from parrots.aa_fortu.aa_fortu_train_multihead_lens import MultiHeadLens
    from parrots.aa_fortu.modules.data_utils import load_text_dataset
    from parrots.cycle_detection import detect_cycles
    
    # Load model
    revision = checkpoint if checkpoint != 'steplatest' else None
    model, tokenizer = load_model_and_tokenizer(model_name, revision, use_bfloat16=False)
    model.eval()
    model.to(device)
    hooked_model = HookedModel(model, layer=f"gpt_neox.layers.{layer}")
    
    # Load lens
    clean_name = model_name.replace('/', '_')
    lens_path = Path(f"/home/mmahaut/projects/parrots/lenses_multihead/{clean_name}/layer_{layer}_multihead_lens.pth")
    lens_data = torch.load(lens_path, weights_only=False, map_location=device)
    lens = MultiHeadLens.from_dict(lens_data)
    lens.to(device)
    
    # Load data
    texts = load_text_dataset(seed=42, n_samples=1000)
    
    # Storage for per-sample contrasts
    original_contrasts = []
    acquired_contrasts = []
    
    # Model config
    num_heads = 8 if 'pythia-70m' in model_name else 16
    
    with torch.no_grad():
        for idx, text in enumerate(texts):
            if idx % 100 == 0:
                print(f"Processing sample {idx}/{len(texts)}")
            
            cat = classification.get(idx, {}).get('category', 'never')
            if cat == 'never':
                continue
            
            # Tokenize and generate
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
            input_ids = inputs['input_ids'].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=False
            )
            
            generated = outputs.sequences[0]
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            
            # Detect cycles
            cycle_info = detect_cycles(generated_text)
            if cycle_info[0] is None:
                continue
            
            # Get attention outputs at cycle start
            cycle_start = cycle_info[1] if len(cycle_info) > 1 else 0
            
            # Get hooked outputs
            layer_outputs = hooked_model.get_outputs()
            if not layer_outputs:
                continue
            
            # Extract attention heads
            for layer_name, layer_data in layer_outputs:
                if isinstance(layer_data, (list, tuple)):
                    layer_data = layer_data[0]
                
                if hasattr(layer_data, 'shape') and len(layer_data.shape) == 3:
                    batch_size, seq_len, hidden_size = layer_data.shape
                    head_dim = hidden_size // num_heads
                    reshaped = layer_data.reshape(batch_size, seq_len, num_heads, head_dim)
                    
                    # Get expected next token
                    expected = generated[min(cycle_start + 1, len(generated) - 1)]
                    
                    # Compute per-head logits at cycle start position
                    pos = min(cycle_start, seq_len - 1)
                    head_contrasts = []
                    
                    for h in range(num_heads):
                        head_act = reshaped[0, pos, h, :]  # [head_dim]
                        head_logits = lens.head_lenses[h](head_act.unsqueeze(0))  # [1, vocab]
                        
                        # Contrast: logit of expected token vs max other
                        expected_logit = head_logits[0, expected].item()
                        head_logits[0, expected] = float('-inf')
                        max_other = head_logits.max().item()
                        contrast = expected_logit - max_other
                        head_contrasts.append(contrast)
                    
                    if cat == 'original':
                        original_contrasts.append(head_contrasts)
                    else:
                        acquired_contrasts.append(head_contrasts)
    
    return np.array(original_contrasts), np.array(acquired_contrasts)

def plot_original_vs_acquired_heads(original_contrasts, acquired_contrasts, 
                                    model_name, checkpoint, layer, output_path):
    """
    Plot attention head contrast difference between original and acquired.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    num_heads = original_contrasts.shape[1] if len(original_contrasts) > 0 else acquired_contrasts.shape[1]
    head_labels = [f'H{i}' for i in range(num_heads)]
    
    # Mean contrasts
    orig_mean = np.mean(original_contrasts, axis=0) if len(original_contrasts) > 0 else np.zeros(num_heads)
    acq_mean = np.mean(acquired_contrasts, axis=0) if len(acquired_contrasts) > 0 else np.zeros(num_heads)
    
    # Plot 1: Side-by-side bars
    x = np.arange(num_heads)
    width = 0.35
    
    axes[0].bar(x - width/2, orig_mean, width, label='Original (innate)', color='#4A90E2')
    axes[0].bar(x + width/2, acq_mean, width, label='Acquired', color='#E74C3C')
    axes[0].set_xlabel('Attention Head')
    axes[0].set_ylabel('Mean Contrast')
    axes[0].set_title('Original vs Acquired Attention Contrasts')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(head_labels)
    axes[0].legend()
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Difference (Acquired - Original)
    diff = acq_mean - orig_mean
    colors = ['#27AE60' if d > 0 else '#E74C3C' for d in diff]
    axes[1].bar(x, diff, color=colors)
    axes[1].set_xlabel('Attention Head')
    axes[1].set_ylabel('Difference (Acquired - Original)')
    axes[1].set_title('Attention Head Difference\n(positive = more active for acquired)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(head_labels)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    
    # Plot 3: Heatmap
    data = np.vstack([orig_mean, acq_mean])
    im = axes[2].imshow(data, aspect='auto', cmap='RdBu_r')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Original', 'Acquired'])
    axes[2].set_xticks(range(num_heads))
    axes[2].set_xticklabels(head_labels)
    axes[2].set_xlabel('Attention Head')
    axes[2].set_title('Attention Contrast Heatmap')
    plt.colorbar(im, ax=axes[2])
    
    model_short = model_name.split('/')[-1]
    fig.suptitle(f'{model_short} - {checkpoint} - Layer {layer}\n'
                 f'(Original: {len(original_contrasts)} samples, Acquired: {len(acquired_contrasts)} samples)',
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='EleutherAI/pythia-70m')
    parser.add_argument('--checkpoint', default='steplatest')
    parser.add_argument('--layer', type=int, default=4)
    parser.add_argument('--output-dir', default='/home/mmahaut/projects/parrots/outputs_multihead_full')
    args = parser.parse_args()
    
    # Load cycle evolution data
    model_short = args.model_name.split('/')[-1]
    clean_model = args.model_name.replace('/', '_')
    cycle_file = Path(f'/home/mmahaut/projects/parrots/cycle_evolution_results/cycle_evolution_status_{clean_model}.json')
    
    if not cycle_file.exists():
        print(f"Cycle evolution file not found: {cycle_file}")
        return
    
    all_status = load_cycle_evolution_data(cycle_file)
    checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
    
    # Classify samples
    classification = classify_samples(all_status, checkpoints)
    
    n_original = sum(1 for c in classification.values() if c['category'] == 'original')
    n_acquired = sum(1 for c in classification.values() if c['category'] == 'acquired')
    n_never = sum(1 for c in classification.values() if c['category'] == 'never')
    print(f"Classification: {n_original} original, {n_acquired} acquired, {n_never} never")
    
    # Run per-sample analysis
    print(f"\nRunning per-sample analysis for {args.model_name} {args.checkpoint} layer {args.layer}")
    original_contrasts, acquired_contrasts = run_per_sample_analysis(
        args.model_name, args.checkpoint, args.layer, classification
    )
    
    print(f"Original samples: {len(original_contrasts)}")
    print(f"Acquired samples: {len(acquired_contrasts)}")
    
    # Plot
    output_path = Path(args.output_dir) / f'original_vs_acquired_heads_{model_short}_{args.checkpoint}_L{args.layer}.png'
    plot_original_vs_acquired_heads(
        original_contrasts, acquired_contrasts,
        args.model_name, args.checkpoint, args.layer,
        str(output_path)
    )

if __name__ == '__main__':
    main()
