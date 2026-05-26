#!/usr/bin/env python3
"""
Logit entropy evolution analysis across all checkpoints.
Computes entropy of output logits at the first token of each cycle,
across cycles 0-4, for multiple model checkpoints.

Creates individual plots per checkpoint showing Natural vs No-Cycle ICL.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from torch.nn.functional import softmax, log_softmax
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from parrots.cycle_detection import detect_cycles
from parrots.aa_fortu.modules.data_utils import load_text_dataset
import argparse
import gc

def compute_entropy(logits, temperature=1.0):
    """Compute entropy of probability distribution from logits"""
    scaled_logits = logits / temperature
    log_probs = log_softmax(scaled_logits, dim=-1)
    probs = softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def extract_entropy_for_cycle(texts, model, tokenizer, n_cycles, batch_size=4, 
                               max_length=64, max_new_tokens=50, device='cuda'):
    """
    Extract entropy data for a specific cycle number.
    Returns dict with 'natural' and 'no_cycle_icl' entropy arrays.
    """
    natural_entropies = []
    no_cycle_icl_entropies = []
    
    # Set padding side for decoder-only models
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Pre-tokenize
    pretokenized = [tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length) for t in texts]
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Cycle {n_cycles}", leave=False):
        batch = pretokenized[i:i+batch_size]
        
        input_ids_list = [b['input_ids'].squeeze(0) for b in batch]
        attention_mask_list = [b['attention_mask'].squeeze(0) for b in batch]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        
        toked = {
            'input_ids': input_ids.long().to(device),
            'attention_mask': attention_mask.long().to(device)
        }
        
        # Generate
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            o1 = model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens,
                               pad_token_id=tokenizer.pad_token_id)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Find prompt lengths
        plengths = toked["attention_mask"].sum(dim=1).tolist()
        
        # Move to CPU for cycle detection
        o1_cpu = [o1[j].detach().cpu() for j in range(len(o1))]
        o1_cpu = [o[o != tokenizer.pad_token_id] for o in o1_cpu]
        
        # Detect cycles
        reps = []
        for o in o1_cpu:
            rep = detect_cycles(o, return_index=True, pad_token_id=tokenizer.pad_token_id)
            reps.append(rep)
        
        # Build inputs for entropy computation
        natural_input = []
        no_cycle_icl_input = []
        
        for j, rep in enumerate(reps):
            if rep[0] is not None:
                # Has cycles - this is "natural" data
                base = o1_cpu[j][:plengths[j] + rep[3]].tolist()
                if n_cycles > 0:
                    cycle_tokens = rep[0] * n_cycles
                    seq = [int(token) for token in (base + cycle_tokens)]
                else:
                    seq = [int(token) for token in base]
                
                if len(seq) > 0:
                    natural_input.append(seq)
            else:
                # No cycles - this is "no_cycle_icl" data
                base = o1_cpu[j][:plengths[j]].tolist()
                if n_cycles > 0:
                    seq = [int(token) for token in (base * n_cycles)]
                else:
                    seq = [int(token) for token in base]
                
                if len(seq) > 0:
                    no_cycle_icl_input.append(seq)
        
        # Helper for batching
        def pad_and_batch(seqs):
            if not seqs:
                return None, None
            non_empty = [s for s in seqs if len(s) > 0]
            if not non_empty:
                return None, None
            max_len = max(len(s) for s in non_empty)
            input_ids = [([tokenizer.pad_token_id] * (max_len - len(s)) + s) for s in non_empty]
            attention_mask = [[0] * (max_len - len(s)) + [1] * len(s) for s in non_empty]
            return torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(attention_mask, dtype=torch.long, device=device)
        
        # Compute entropy for natural data
        if natural_input:
            nat_ids, nat_mask = pad_and_batch(natural_input)
            if nat_ids is not None and nat_ids.shape[0] > 0:
                with torch.no_grad():
                    outputs = model(**{'input_ids': nat_ids, 'attention_mask': nat_mask})
                    logits = outputs.logits[:, -1, :].float()
                    entropy = compute_entropy(logits)
                    natural_entropies.extend(entropy.cpu().numpy().tolist())
        
        # Compute entropy for no-cycle ICL data
        if no_cycle_icl_input:
            nc_ids, nc_mask = pad_and_batch(no_cycle_icl_input)
            if nc_ids is not None and nc_ids.shape[0] > 0:
                with torch.no_grad():
                    outputs = model(**{'input_ids': nc_ids, 'attention_mask': nc_mask})
                    logits = outputs.logits[:, -1, :].float()
                    entropy = compute_entropy(logits)
                    no_cycle_icl_entropies.extend(entropy.cpu().numpy().tolist())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    tokenizer.padding_side = original_padding_side
    
    return {
        'natural': np.array(natural_entropies),
        'no_cycle_icl': np.array(no_cycle_icl_entropies)
    }

def run_entropy_analysis_for_checkpoint(model_name, revision, texts, n_samples=200, device='cuda'):
    """
    Run entropy analysis across cycles 0-4 for a single checkpoint.
    """
    print(f"\nLoading model: {model_name} (revision: {revision or 'latest'})")
    
    # Load model
    if revision:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    results = {
        'cycle': [],
        'data_type': [],
        'entropy_values': []
    }
    
    # Process each cycle
    for cycle in range(5):
        print(f"  Processing cycle {cycle}...")
        
        entropy_data = extract_entropy_for_cycle(
            texts[:n_samples], model, tokenizer, 
            n_cycles=cycle, batch_size=4,
            max_length=64, max_new_tokens=50, device=device
        )
        
        # Store results
        for data_type, entropies in entropy_data.items():
            if len(entropies) > 0:
                results['cycle'].extend([cycle] * len(entropies))
                results['data_type'].extend([data_type] * len(entropies))
                results['entropy_values'].extend(entropies.tolist())
                print(f"    {data_type}: {len(entropies)} samples, mean={np.mean(entropies):.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return pd.DataFrame(results)

def create_checkpoint_entropy_plot(df, checkpoint_name, model_name, output_path):
    """
    Create entropy evolution boxplot for a single checkpoint.
    """
    # Filter out ICL data if present
    df_filtered = df[df['data_type'] != 'icl'].copy()
    
    if df_filtered.empty:
        print(f"  No data to plot for {checkpoint_name}")
        return
    
    # Paper-ready styling
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Boxplot with hue for paired layout
    palette = {'natural': '#e74c3c', 'no_cycle_icl': '#3498db'}
    
    # Check which data types are present
    present_types = df_filtered['data_type'].unique()
    palette_filtered = {k: v for k, v in palette.items() if k in present_types}
    
    if len(palette_filtered) > 0:
        sns.boxplot(data=df_filtered, x='cycle', y='entropy_values', hue='data_type', 
                    ax=ax, palette=palette_filtered)
        
        ax.set_xlabel('Cycle Number', fontsize=12)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        
        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        label_map = {'natural': 'Natural', 'no_cycle_icl': 'No-Cycle ICL'}
        new_labels = [label_map.get(l, l) for l in labels]
        ax.legend(handles, new_labels, fontsize=10, frameon=True, framealpha=0.9)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Compact title
        short_model = model_name.split('/')[-1]
        ax.set_title(f'{short_model} @ {checkpoint_name}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Logit entropy analysis across checkpoints")
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="./outputs_entropy_evolution")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define checkpoints based on model
    if 'pythia' in args.model_name.lower():
        checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', None]  # None = steplatest
        checkpoint_names = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
    elif 'olmo' in args.model_name.lower():
        checkpoints = ['step1000-tokens4B', 'step343000-tokens1438B', 'step425000-tokens1781B', 
                      'step509000-tokens2134B', 'step593000-tokens2486B', 'step738020-tokens3094B']
        checkpoint_names = checkpoints
    else:
        # Default
        checkpoints = [None]
        checkpoint_names = ['latest']
    
    # Create output directory
    safe_model = args.model_name.replace('/', '_')
    output_dir = Path(args.output_dir) / safe_model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load texts once
    print(f"Loading {args.n_samples} text samples...")
    texts = load_text_dataset(seed=args.seed, n_samples=args.n_samples)
    
    # Process each checkpoint
    all_results = {}
    
    for revision, cp_name in zip(checkpoints, checkpoint_names):
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {cp_name}")
        print(f"{'='*60}")
        
        df = run_entropy_analysis_for_checkpoint(
            args.model_name, revision, texts,
            n_samples=args.n_samples, device=device
        )
        
        # Save CSV
        csv_path = output_dir / f"entropy_results_{cp_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")
        
        # Create plot
        plot_path = output_dir / f"logit_entropy_evolution_{cp_name}.png"
        create_checkpoint_entropy_plot(df, cp_name, args.model_name, plot_path)
        
        all_results[cp_name] = df
    
    # Create summary comparison plot
    print("\n" + "="*60)
    print("Creating summary comparison plot...")
    print("="*60)
    
    # Combine all data with checkpoint labels
    combined_df = []
    for cp_name, df in all_results.items():
        df = df.copy()
        df['checkpoint'] = cp_name
        combined_df.append(df)
    
    if combined_df:
        combined_df = pd.concat(combined_df, ignore_index=True)
        combined_csv = output_dir / "entropy_results_all_checkpoints.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined CSV: {combined_csv}")
    
    print("\n✅ All entropy analyses complete!")
    print(f"Results in: {output_dir}")

if __name__ == "__main__":
    main()
