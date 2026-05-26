#!/usr/bin/env python3
"""
Multihead analysis: Original vs Acquired attention head contrast evolution.

Creates line plots (similar to multihead_cycle_evolution.png) showing how 
attention head contrasts evolve across training checkpoints, split by 
original (innate) vs acquired repetition samples.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import typer
import numpy as np
from pathlib import Path
from parrots.aa_fortu.modules.model_utils import HookedModel, load_model_and_tokenizer, get_device
from parrots.aa_fortu.aa_fortu_train_multihead_lens import MultiHeadLens
from parrots.aa_fortu.modules.data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles
import torch
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from typing import Optional


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


def extract_contrasts_for_checkpoint(
    text, hooked_model, tokenizer, lens, classification,
    batch_size=8, max_length=32, max_new_tokens=1000,
    layer=None, device=None, n_cycle_iterations=6, sample_index_offset=0
):
    """Extract attention head contrasts, split by original vs acquired category.
    
    Computes contrasts at multiple positions within the cycle (n_cycle_iterations).
    Only analyzes positions within the cycle (post-prompt), not the prompt itself.
    
    Returns contrasts organized by cycle iteration number for evolution plots.
    """
    if device is None:
        device = get_device()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Organize by: category -> iteration -> head -> list of contrasts
    original_contrasts = {i: {} for i in range(n_cycle_iterations)}
    acquired_contrasts = {i: {} for i in range(n_cycle_iterations)}
    
    stats = {
        'original_total': 0, 'original_repeating': 0,
        'acquired_total': 0, 'acquired_repeating': 0,
    }
    
    # Pre-tokenize
    pretokenized = [tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length) for t in text]
    
    for i in tqdm(range(0, len(text), batch_size), desc="Processing samples", leave=False):
        batch = pretokenized[i:i+batch_size]
        batch_indices = list(range(sample_index_offset + i, sample_index_offset + min(i + batch_size, len(text))))
        
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
            hooked_model.clear()
            continue
        
        # Process each sample in batch
        for j, (sample_idx, rep, plen) in enumerate(zip(batch_indices, reps, plengths)):
            cat = classification.get(sample_idx, 'never')
            
            if cat == 'never':
                continue
            elif cat == 'original':
                stats['original_total'] += 1
            else:
                stats['acquired_total'] += 1
            
            has_cycle = rep[0] is not None
            
            if cat == 'original' and has_cycle:
                stats['original_repeating'] += 1
            elif cat == 'acquired' and has_cycle:
                stats['acquired_repeating'] += 1
            
            if not has_cycle:
                continue
            
            # Get cycle info
            cycle_start = rep[3]  # relative to generated sequence
            cycle_tokens = rep[0]  # the repeating pattern
            cycle_size = rep[1]  # length of pattern
            cycle_count = rep[2]  # how many times it repeated
            
            # Get logits
            all_logits = hooked_model.get_all_logits()
            if all_logits is None:
                continue
            
            # Determine how many iterations we can actually compute
            # (limited by cycle_count and n_cycle_iterations parameter)
            max_iterations = min(n_cycle_iterations, cycle_count)
            
            # Iterate over multiple positions in the cycle
            for iter_idx in range(max_iterations):
                # Position in the full sequence (plen = prompt length, cycle_start relative to generated)
                pos = plen + cycle_start + iter_idx * cycle_size
                
                if pos >= all_logits.shape[1]:
                    break
                
                # Expected token cycles through the pattern
                expected_token = cycle_tokens[0]  # first token of the cycle at each cycle start
                
                logits_at_pos = all_logits[j, pos, :]
                
                # Check if prediction matches
                top_token = logits_at_pos.argmax().item()
                if top_token != expected_token:
                    continue
                
                # Get contrast token (second highest)
                logits_copy = logits_at_pos.clone()
                logits_copy[expected_token] = float('-inf')
                contrast_token = logits_copy.argmax().item()
                
                # Process attention outputs
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
                        num_heads = hidden_size // 64
                    
                    head_dim = hidden_size // num_heads
                    reshaped = layer_data.reshape(batch_size_actual, seq_len, num_heads, head_dim)
                    attn_pos = min(pos, seq_len - 1)
                    
                    layer_idx = int(re.search(r'\d+', layer_name).group())
                    if layer_idx not in lens:
                        continue
                    layer_lens = lens[layer_idx]
                    
                    for h in range(num_heads):
                        head_act = reshaped[j, attn_pos, h, :]
                        head_logits = layer_lens.head_lenses[h](head_act.unsqueeze(0).float())
                        
                        expected_logit = head_logits[0, expected_token].item()
                        contrast_logit = head_logits[0, contrast_token].item()
                        contrast = expected_logit - contrast_logit
                        
                        key = f"L{layer_idx}_H{h}"
                        
                        if cat == 'original':
                            original_contrasts[iter_idx].setdefault(key, []).append(contrast)
                        else:
                            acquired_contrasts[iter_idx].setdefault(key, []).append(contrast)
        
        hooked_model.clear()
    
    return original_contrasts, acquired_contrasts, stats


def plot_cycle_iteration_grid(results_df, model_name, layer_label, output_path, checkpoints, max_heads_to_show=8):
    """
    Plot cycle-iteration evolution with checkpoints stacked in rows.
    - Rows: checkpoints
    - Columns: Innate (Original) vs Acquired
    - X-axis: cycle iteration (0..N)
    - Y-axis: contrast
    """
    if results_df.empty:
        print("No data to plot")
        return

    iterations = sorted(results_df['iteration'].unique())

    head_variances = results_df.groupby('head')['contrast'].var().sort_values(ascending=False)
    interesting_heads = head_variances.head(max_heads_to_show).index.tolist()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    head_color_map = {head: colors[i % len(colors)] for i, head in enumerate(interesting_heads)}
    head_style_map = {head: styles[i % len(styles)] for i, head in enumerate(interesting_heads)}

    y_min = results_df['contrast'].min()
    y_max = results_df['contrast'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1 if y_range > 0 else 0.1
    y_min_padded = y_min - y_padding
    y_max_padded = y_max + y_padding

    n_rows = len(checkpoints)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows), sharex=True, sharey=True)

    if n_rows == 1:
        axes = np.array([axes])

    def checkpoint_label(cp):
        label = cp.replace('step', '')
        if label.isdigit():
            num = int(label)
            if num >= 1000:
                label = f"{num // 1000}K"
        elif label == 'latest':
            label = 'latest'
        return label

    background_alpha = 0.08

    for row_idx, checkpoint in enumerate(checkpoints):
        row_label = checkpoint_label(checkpoint)
        for col_idx, (category, title, border_color, bg_color) in enumerate([
            ('original', 'Innate (Original)', '#4A90E2', '#e8f4fc'),
            ('acquired', 'Acquired', '#E74C3C', '#fce8e8')
        ]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(bg_color)

            cat_df = results_df[(results_df['checkpoint'] == checkpoint) & (results_df['category'] == category)]

            if cat_df.empty:
                ax.set_title(f'{row_label} - {title}\n(no data)', fontsize=12, fontweight='bold')
                continue

            for head in cat_df['head'].unique():
                head_data = cat_df[cat_df['head'] == head].sort_values('iteration')
                ax.plot(head_data['iteration'], head_data['contrast'],
                        color='lightgrey', alpha=background_alpha, linewidth=1)

            interesting_df = cat_df[cat_df['head'].isin(interesting_heads)]
            if not interesting_df.empty:
                for head in interesting_heads:
                    head_data = interesting_df[interesting_df['head'] == head].sort_values('iteration')
                    if not head_data.empty:
                        ax.plot(head_data['iteration'], head_data['contrast'],
                                color=head_color_map[head], linestyle=head_style_map[head],
                                marker='o', markersize=6, linewidth=2, label=head)

            ax.set_title(f'{row_label} - {title}', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=border_color, alpha=0.2))
            ax.set_xticks(iterations)
            ax.set_xticklabels([str(i) for i in iterations], fontsize=9)
            ax.set_ylim(y_min_padded, y_max_padded)
            ax.grid(True, alpha=0.3)

            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color(border_color)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Cycle Iteration", fontsize=10)

            if col_idx == 0:
                ax.set_ylabel("Contrast (expected - alternative)", fontsize=10)

    if interesting_heads:
        legend_elements = [
            plt.Line2D([0], [0], color=head_color_map[head],
                       linestyle=head_style_map[head], marker='o',
                       markersize=5, linewidth=2, label=head)
            for head in interesting_heads
        ]
        ncol = min(len(legend_elements), 4)
        fig.legend(handles=legend_elements, loc='lower center', ncol=ncol,
                   bbox_to_anchor=(0.5, -0.02), fontsize=10)

    model_short = model_name.split('/')[-1]
    fig.suptitle(f'{model_short} {layer_label}: Cycle Iteration Evolution\n'
                 f'Checkpoints stacked by row (Innate vs Acquired)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.90)

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main(
    model_name: str = "EleutherAI/pythia-70m",
    layer: int = 4,
    all_layers: bool = False,
    seed: int = 42,
    batch_size: int = 8,
    max_length: int = 32,
    max_new_tokens: int = 1000,
    n_samples: int = 300,
    total_samples: int = 300,
    chunk_index: Optional[int] = None,
    n_cycle_iterations: int = 6,
    output_dir: str = "/home/mmahaut/projects/parrots/outputs_multihead_full",
    checkpoints_to_run: str = "step1,step1000,step10000,step100000,steplatest",
    checkpoint: Optional[str] = None
):
    """Run multihead analysis across checkpoints, stacking rows per checkpoint.
    
    X-axis = cycle iteration (0..N)
    Y-axis = contrast
    Rows = checkpoints
    Columns = Innate vs Acquired
    """
    
    all_checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']

    def sort_checkpoint_key(cp):
        if cp == 'steplatest':
            return float('inf')
        if cp.startswith('step'):
            try:
                return int(cp[4:])
            except ValueError:
                return 0
        return 0

    if checkpoint is not None:
        run_checkpoints = [checkpoint]
    else:
        run_checkpoints = [cp.strip() for cp in checkpoints_to_run.split(',') if cp.strip()]
    run_checkpoints = sorted(run_checkpoints, key=sort_checkpoint_key)

    if all_layers:
        layer = None
    
    # Load classification
    clean_model = model_name.replace('/', '_')
    cycle_file = Path(f'/home/mmahaut/projects/parrots/cycle_evolution_results/cycle_evolution_status_{clean_model}.json')
    
    if not cycle_file.exists():
        print(f"ERROR: Cycle evolution file not found: {cycle_file}")
        return
    
    all_status = load_cycle_evolution_data(cycle_file)
    classification = classify_samples(all_status, all_checkpoints)
    
    n_original = sum(1 for c in classification.values() if c == 'original')
    n_acquired = sum(1 for c in classification.values() if c == 'acquired')
    print(f"Classification: {n_original} original, {n_acquired} acquired")
    
    # Model config
    if 'pythia-70m' in model_name:
        num_heads = 8
    elif 'pythia-1.4b' in model_name:
        num_heads = 16
    else:
        num_heads = 16
    
    # Load multihead lenses
    lens_path = Path(f"/home/mmahaut/projects/parrots/lenses_multihead/{clean_model}")
    if not lens_path.exists():
        print(f"ERROR: Lens path not found: {lens_path}")
        return
    
    device = get_device()
    lens_paths = sorted(lens_path.glob("*.pth"), key=lambda x: int(re.search(r'\d+', x.stem).group()))
    lens = {}
    for p in lens_paths:
        layer_idx = int(re.search(r'\d+', p.stem).group())
        lens_data = torch.load(p, weights_only=False, map_location=device)
        lens[layer_idx] = MultiHeadLens.from_dict(lens_data)
        lens[layer_idx].to(device)
    print(f"Loaded {len(lens)} multihead lenses")
    
    # Load data (optionally chunked from a larger deterministic sample)
    if chunk_index is not None:
        texts_full = load_text_dataset(seed=seed, n_samples=total_samples)
        start_idx = chunk_index * n_samples
        end_idx = min(start_idx + n_samples, total_samples)
        if start_idx >= total_samples:
            print(f"Chunk index {chunk_index} out of range for total_samples={total_samples}")
            return
        texts = texts_full[start_idx:end_idx]
        sample_index_offset = start_idx
        print(f"Using chunk {chunk_index}: samples {start_idx}..{end_idx - 1}")
    else:
        texts = load_text_dataset(seed=seed, n_samples=n_samples)
        sample_index_offset = 0
    
    # Collect results across checkpoints
    all_results = []
    
    for checkpoint in run_checkpoints:
        print(f"\n=== Processing {checkpoint} ===")
        
        revision = checkpoint if checkpoint != 'steplatest' else None
        
        # Load model for this checkpoint
        model, tokenizer = load_model_and_tokenizer(model_name, revision, use_bfloat16=False)
        model.eval()
        model.to(device)
        
        hook_layer = None if layer is None else f"gpt_neox.layers.{layer}"
        hooked_model = HookedModel(model, layer=hook_layer)
        
        # Run analysis
        original_contrasts, acquired_contrasts, stats = extract_contrasts_for_checkpoint(
            texts, hooked_model, tokenizer, lens, classification,
            batch_size=batch_size, max_length=max_length,
            max_new_tokens=max_new_tokens, layer=layer, device=device,
            n_cycle_iterations=n_cycle_iterations, sample_index_offset=sample_index_offset
        )
        
        print(f"  Original: {stats['original_repeating']}/{stats['original_total']} repeating")
        print(f"  Acquired: {stats['acquired_repeating']}/{stats['acquired_total']} repeating")
        
        # Aggregate results - now organized by iteration
        for iter_idx, head_contrasts in original_contrasts.items():
            for head_key, contrasts in head_contrasts.items():
                if contrasts:  # only add if we have data
                    all_results.append({
                        'checkpoint': checkpoint,
                        'iteration': iter_idx,
                        'head': head_key,
                        'category': 'original',
                        'contrast': np.mean(contrasts),
                        'n_samples': len(contrasts)
                    })
        
        for iter_idx, head_contrasts in acquired_contrasts.items():
            for head_key, contrasts in head_contrasts.items():
                if contrasts:  # only add if we have data
                    all_results.append({
                        'checkpoint': checkpoint,
                        'iteration': iter_idx,
                        'head': head_key,
                        'category': 'acquired',
                        'contrast': np.mean(contrasts),
                        'n_samples': len(contrasts)
                    })
        
        # Clean up
        del model, hooked_model
        torch.cuda.empty_cache()
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save CSV
    model_short = model_name.split('/')[-1]
    layer_tag = "all_layers" if layer is None else f"L{layer}"
    chunk_tag = f"_chunk{chunk_index}" if chunk_index is not None else ""
    if len(run_checkpoints) == 1:
        csv_path = Path(output_dir) / f"cycle_iteration_evolution_{model_short}_{layer_tag}_{run_checkpoints[0]}{chunk_tag}.csv"
    else:
        csv_path = Path(output_dir) / f"cycle_iteration_grid_{model_short}_{layer_tag}{chunk_tag}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved data: {csv_path}")
    
    # Generate plot
    if chunk_index is None:
        if len(run_checkpoints) == 1:
            output_path = Path(output_dir) / f"cycle_iteration_evolution_{model_short}_{layer_tag}_{run_checkpoints[0]}.png"
        else:
            output_path = Path(output_dir) / f"cycle_iteration_grid_{model_short}_{layer_tag}.png"
        layer_label = "All layers" if layer is None else f"Layer {layer}"
        plot_cycle_iteration_grid(results_df, model_name, layer_label, str(output_path), run_checkpoints)
    
    print("\nDone!")


if __name__ == "__main__":
    typer.run(main)
