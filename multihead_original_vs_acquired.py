#!/usr/bin/env python3
"""
Modified multihead analysis pipeline that computes SEPARATE attention head
activation heatmaps for original (innate) vs acquired repetition samples.

Uses the existing multihead lenses and contrast analysis, but splits results
by sample classification.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import typer
import numpy as np
from pathlib import Path
from parrots.aa_fortu.modules.model_utils import HookedModel, load_model_and_tokenizer, get_device
from parrots.aa_fortu.aa_fortu_train_lens import Lens
from parrots.aa_fortu.aa_fortu_train_multihead_lens import MultiHeadLens
from parrots.aa_fortu.modules.data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import re
import matplotlib.pyplot as plt


def load_cycle_evolution_data(results_file):
    """Load cycle evolution JSON data"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return {cp: {int(k): v for k, v in status.items()} 
            for cp, status in data.items()}


def classify_samples(all_status, checkpoints):
    """Classify samples as original, acquired, or never-repeating."""
    n_texts = max(max(status.keys()) for status in all_status.values() if status) + 1
    
    classification = {}
    for idx in range(n_texts):
        first_rep_cp = None
        for cp in checkpoints:
            if cp in all_status and idx in all_status[cp]:
                if all_status[cp][idx]:
                    first_rep_cp = cp
                    break
        
        if first_rep_cp is None:
            classification[idx] = 'never'
        elif first_rep_cp == checkpoints[0]:
            classification[idx] = 'original'
        else:
            classification[idx] = 'acquired'
    
    return classification


def extract_contrasts_by_category(
    text, hooked_model, tokenizer, lens, classification,
    n_cycles=0, batch_size=8, max_length=32, max_new_tokens=1000,
    layer=None, device=None
):
    """
    Extract attention head contrasts, split by original vs acquired category.
    
    Returns:
        original_contrasts: dict of {layer_head: [contrast_values]} for original samples
        acquired_contrasts: dict of {layer_head: [contrast_values]} for acquired samples
        stats: dict with counts
    """
    if device is None:
        device = get_device()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    original_contrasts = {}
    acquired_contrasts = {}
    
    stats = {
        'original_total': 0, 'original_repeating': 0,
        'acquired_total': 0, 'acquired_repeating': 0,
        'never_total': 0
    }
    
    # Pre-tokenize
    pretokenized = [tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length) for t in text]
    
    for i in tqdm(range(0, len(text), batch_size), desc="Processing samples"):
        batch = pretokenized[i:i+batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(text))))
        
        input_ids_list = [b['input_ids'].squeeze(0) for b in batch]
        attention_mask_list = [b['attention_mask'].squeeze(0) for b in batch]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        toked = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}
        
        # Generate
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
            o1 = hooked_model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens)
        
        # Get prompt lengths
        plengths = toked["attention_mask"].sum(dim=1).tolist()
        
        # CPU for cycle detection
        o1_cpu = [o1[j].detach().cpu() for j in range(len(o1))]
        o1_cpu = [o[o != tokenizer.pad_token_id] for o in o1_cpu]
        
        # Detect cycles
        reps = []
        for o in o1_cpu:
            rep = detect_cycles(o[plengths[0]:], return_index=True, pad_token_id=tokenizer.pad_token_id)
            reps.append(rep)
        
        # Get attention outputs
        attn_outputs = hooked_model.attn_outputs
        if not attn_outputs:
            continue
        
        # Process each sample in batch
        for j, (sample_idx, rep, plen) in enumerate(zip(batch_indices, reps, plengths)):
            cat = classification.get(sample_idx, 'never')
            
            if cat == 'never':
                stats['never_total'] += 1
                continue
            elif cat == 'original':
                stats['original_total'] += 1
            else:
                stats['acquired_total'] += 1
            
            has_cycle = rep[0] is not None
            
            if cat == 'original':
                if has_cycle:
                    stats['original_repeating'] += 1
            else:
                if has_cycle:
                    stats['acquired_repeating'] += 1
            
            if not has_cycle:
                continue
            
            # Get cycle start position (relative to generated, so add plen for absolute position)
            cycle_start = rep[3]
            expected_token = rep[0][0]
            
            # Get logits at cycle start (plen + cycle_start = absolute position)
            all_logits = hooked_model.get_all_logits()
            if all_logits is None:
                continue
            
            pos = min(plen + cycle_start, all_logits.shape[1] - 1)
            logits_at_pos = all_logits[j, pos, :]  # [vocab]
            
            # Check if prediction matches
            top_token = logits_at_pos.argmax().item()
            if top_token != expected_token:
                continue
            
            # Get contrast token (second highest)
            logits_at_pos[expected_token] = float('-inf')
            contrast_token = logits_at_pos.argmax().item()
            logits_at_pos[expected_token] = all_logits[j, pos, expected_token]  # restore
            
            # Process attention outputs for this sample
            for layer_name, layer_data in attn_outputs:
                if layer is not None and str(layer) not in layer_name:
                    continue
                
                if isinstance(layer_data, (list, tuple)):
                    layer_data = layer_data[0]
                
                if not hasattr(layer_data, 'shape') or len(layer_data.shape) != 3:
                    continue
                
                batch_size_actual, seq_len, hidden_size = layer_data.shape
                
                # Determine num_heads
                if hidden_size == 512:
                    num_heads = 8
                elif hidden_size == 2048:
                    num_heads = 16
                elif hidden_size == 4096:
                    num_heads = 32
                else:
                    num_heads = hidden_size // 64  # fallback
                
                head_dim = hidden_size // num_heads
                
                # Reshape to [batch, seq, heads, head_dim]
                reshaped = layer_data.reshape(batch_size_actual, seq_len, num_heads, head_dim)
                
                # Get attention at cycle start position
                attn_pos = min(pos, seq_len - 1)
                
                # Get lens for this layer
                layer_idx = int(re.search(r'\d+', layer_name).group())
                if layer_idx not in lens:
                    continue
                layer_lens = lens[layer_idx]
                
                # Compute contrast for each head
                for h in range(num_heads):
                    head_act = reshaped[j, attn_pos, h, :]  # [head_dim]
                    head_logits = layer_lens.head_lenses[h](head_act.unsqueeze(0).float())  # [1, vocab]
                    
                    expected_logit = head_logits[0, expected_token].item()
                    contrast_logit = head_logits[0, contrast_token].item()
                    contrast = expected_logit - contrast_logit
                    
                    key = f"{layer_name}_head_{h}"
                    
                    if cat == 'original':
                        original_contrasts.setdefault(key, []).append(contrast)
                    else:
                        acquired_contrasts.setdefault(key, []).append(contrast)
        
        # Clear outputs for next batch
        hooked_model.clear()
    
    return original_contrasts, acquired_contrasts, stats


def plot_original_vs_acquired_heatmaps(original_contrasts, acquired_contrasts, 
                                        num_heads, model_name, checkpoint, layer, output_path):
    """Plot attention head activation comparison between original and acquired."""
    
    # Compute means
    original_means = np.zeros(num_heads)
    acquired_means = np.zeros(num_heads)
    
    for h in range(num_heads):
        key = f"gpt_neox.layers.{layer}_head_{h}"
        if key in original_contrasts and original_contrasts[key]:
            original_means[h] = np.mean(original_contrasts[key])
        if key in acquired_contrasts and acquired_contrasts[key]:
            acquired_means[h] = np.mean(acquired_contrasts[key])
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
    })
    
    head_labels = [f'H{i}' for i in range(num_heads)]
    x = np.arange(num_heads)
    width = 0.35
    
    # Plot 1: Side-by-side bars
    ax = axes[0]
    ax.bar(x - width/2, original_means, width, label='Original (innate)', color='#4A90E2', alpha=0.8)
    ax.bar(x + width/2, acquired_means, width, label='Acquired', color='#E74C3C', alpha=0.8)
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Mean Contrast (expected - alternative)')
    ax.set_title('Attention Head Activation by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Difference (Acquired - Original)
    ax = axes[1]
    diff = acquired_means - original_means
    colors = ['#27AE60' if d > 0 else '#E74C3C' for d in diff]
    ax.bar(x, diff, color=colors)
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Difference (Acquired - Original)')
    ax.set_title('Per-Head Difference\n(positive = stronger for acquired)')
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels)
    ax.axhline(y=0, color='gray', linestyle='--')
    
    # Plot 3: Heatmap
    ax = axes[2]
    data = np.vstack([original_means, acquired_means])
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Original', 'Acquired'])
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels(head_labels)
    ax.set_xlabel('Attention Head')
    ax.set_title('Contrast Heatmap')
    plt.colorbar(im, ax=ax, label='Contrast')
    
    # Get sample counts
    n_orig = len(next(iter(original_contrasts.values()))) if original_contrasts else 0
    n_acq = len(next(iter(acquired_contrasts.values()))) if acquired_contrasts else 0
    
    model_short = model_name.split('/')[-1]
    fig.suptitle(f'{model_short} - {checkpoint} - Layer {layer}\n'
                 f'(Original: {n_orig} samples, Acquired: {n_acq} samples)',
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return original_means, acquired_means


def main(
    model_name: str = "EleutherAI/pythia-70m",
    revision: str = None,
    layer: int = 4,
    n_cycles: int = 0,
    seed: int = 42,
    batch_size: int = 8,
    max_length: int = 32,
    max_new_tokens: int = 1000,
    n_samples: int = 300,
    output_dir: str = "/home/mmahaut/projects/parrots/outputs_multihead_full"
):
    """Run multihead analysis split by original vs acquired."""
    
    checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
    
    # Load classification
    clean_model = model_name.replace('/', '_')
    cycle_file = Path(f'/home/mmahaut/projects/parrots/cycle_evolution_results/cycle_evolution_status_{clean_model}.json')
    
    if not cycle_file.exists():
        print(f"ERROR: Cycle evolution file not found: {cycle_file}")
        return
    
    all_status = load_cycle_evolution_data(cycle_file)
    classification = classify_samples(all_status, checkpoints)
    
    n_original = sum(1 for c in classification.values() if c == 'original')
    n_acquired = sum(1 for c in classification.values() if c == 'acquired')
    print(f"Classification: {n_original} original, {n_acquired} acquired")
    
    # Load model
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, revision, use_bfloat16=False)
    model.eval()
    model.to(device)
    
    # Determine num_heads
    if 'pythia-70m' in model_name:
        num_heads = 8
        max_layers = 6
    elif 'pythia-1.4b' in model_name:
        num_heads = 16
        max_layers = 24
    else:
        num_heads = 16
        max_layers = 24
    
    hooked_model = HookedModel(model, layer=f"gpt_neox.layers.{layer}")
    print(f"Loaded {model_name} with {len(hooked_model.hooks)} hooks")
    
    # Load multihead lens
    lens_path = Path(f"/home/mmahaut/projects/parrots/lenses_multihead/{clean_model}")
    if not lens_path.exists():
        print(f"ERROR: Lens path not found: {lens_path}")
        return
    
    lens_paths = sorted(lens_path.glob("*.pth"), key=lambda x: int(re.search(r'\d+', x.stem).group()))
    lens = {}
    for p in lens_paths:
        layer_idx = int(re.search(r'\d+', p.stem).group())
        lens_data = torch.load(p, weights_only=False, map_location=device)
        lens[layer_idx] = MultiHeadLens.from_dict(lens_data)
        lens[layer_idx].to(device)
    print(f"Loaded {len(lens)} multihead lenses")
    
    # Load data
    texts = load_text_dataset(seed=seed, n_samples=n_samples)
    
    # Run analysis
    print(f"\nRunning analysis for {model_name} layer {layer}...")
    original_contrasts, acquired_contrasts, stats = extract_contrasts_by_category(
        texts, hooked_model, tokenizer, lens, classification,
        n_cycles=n_cycles, batch_size=batch_size, max_length=max_length,
        max_new_tokens=max_new_tokens, layer=layer, device=device
    )
    
    print(f"\nStats:")
    print(f"  Original: {stats['original_repeating']}/{stats['original_total']} repeating")
    print(f"  Acquired: {stats['acquired_repeating']}/{stats['acquired_total']} repeating")
    
    # Generate plot
    checkpoint = revision if revision else 'steplatest'
    model_short = model_name.split('/')[-1]
    output_path = Path(output_dir) / f"original_vs_acquired_heads_{model_short}_{checkpoint}_L{layer}.png"
    
    original_means, acquired_means = plot_original_vs_acquired_heatmaps(
        original_contrasts, acquired_contrasts,
        num_heads, model_name, checkpoint, layer, str(output_path)
    )
    
    # Print numerical results
    print(f"\nOriginal mean contrasts: {original_means}")
    print(f"Acquired mean contrasts: {acquired_means}")
    print(f"Difference (acq - orig): {acquired_means - original_means}")


if __name__ == "__main__":
    typer.run(main)
