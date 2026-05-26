#!/usr/bin/env python3
"""
Entropy analysis with proper 3-way categorization matching the alluvial plot:
1. Natural repeating - samples that repeat without ICL prompting
2. ICL-induced - samples that don't repeat naturally BUT DO repeat with ICL
3. Never repeating - samples that don't repeat even with ICL

This requires running BOTH natural and ICL conditions to properly categorize.
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import typer
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from parrots.aa_fortu.modules.data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles


def compute_entropy(logits):
    """Compute entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def run_generation(model, tokenizer, texts, max_length=32, max_new_tokens=1000, 
                   batch_size=8, device='cuda', use_icl=False):
    """
    Run model generation on texts.
    
    Args:
        use_icl: If True, use ICL prompting (text repeated as prompt)
    
    Returns:
        List of (text_idx, has_cycle, cycle_info, output_tokens) tuples
    """
    results = []
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"{'ICL' if use_icl else 'Natural'} generation"):
        batch_texts = texts[i:i+batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(texts))))
        
        # Tokenize
        if use_icl:
            # ICL: repeat text as prompt (simulate ICL context)
            icl_texts = [f"{t} {t}" for t in batch_texts]  # Simple ICL: repeat text
            encoded = [tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length*2) 
                      for t in icl_texts]
        else:
            # Natural: just the text
            encoded = [tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length) 
                      for t in batch_texts]
        
        input_ids_list = [e['input_ids'].squeeze(0) for e in encoded]
        attention_mask_list = [e['attention_mask'].squeeze(0) for e in encoded]
        
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Get prompt lengths
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        
        # Detect cycles for each output
        for j, (idx, plen) in enumerate(zip(batch_indices, prompt_lengths)):
            output = outputs[j].cpu()
            output = output[output != tokenizer.pad_token_id]  # Remove padding
            generated = output[plen:]  # Get generated part only
            
            cycle_info = detect_cycles(generated, return_index=True, pad_token_id=tokenizer.pad_token_id)
            has_cycle = cycle_info[0] is not None
            
            results.append((idx, has_cycle, cycle_info, output.tolist()))
    
    return results


def compute_entropy_for_sequences(model, tokenizer, sequences, device='cuda', batch_size=8):
    """Compute entropy at the last position for each sequence."""
    entropies = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        
        # Pad and batch
        max_len = max(len(s) for s in batch)
        input_ids = [[tokenizer.pad_token_id] * (max_len - len(s)) + s for s in batch]
        attention_mask = [[0] * (max_len - len(s)) + [1] * len(s) for s in batch]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].float()
            entropy = compute_entropy(logits)
            entropies.extend(entropy.cpu().numpy().tolist())
    
    return entropies


def categorize_samples(natural_results, icl_results):
    """
    Categorize samples into 3 categories:
    - natural: cycles naturally
    - icl_induced: doesn't cycle naturally, but DOES cycle with ICL
    - never: doesn't cycle in either condition
    """
    categories = {}
    
    # Build index maps
    natural_cycles = {idx: has_cycle for idx, has_cycle, _, _ in natural_results}
    icl_cycles = {idx: has_cycle for idx, has_cycle, _, _ in icl_results}
    
    for idx in natural_cycles:
        if natural_cycles[idx]:
            categories[idx] = 'natural'
        elif idx in icl_cycles and icl_cycles[idx]:
            categories[idx] = 'icl_induced'
        else:
            categories[idx] = 'never'
    
    return categories


def run_entropy_analysis(
    model_name: str,
    revision: str,
    texts: list,
    n_cycles: int = 2,
    batch_size: int = 8,
    max_length: int = 32,
    max_new_tokens: int = 1000,
    device: str = 'cuda'
):
    """
    Run complete entropy analysis with proper 3-way categorization.
    
    Returns dict with:
    - categories: {idx: category} mapping
    - entropies: {category: [entropy_values]}
    - counts: {category: count}
    """
    print(f"\nLoading model {model_name} {revision or 'latest'}...")
    model, tokenizer = load_model_and_tokenizer(model_name, revision)
    model.eval()
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run natural generation
    print("\n1. Running NATURAL generation...")
    natural_results = run_generation(
        model, tokenizer, texts, max_length, max_new_tokens, batch_size, device, use_icl=False
    )
    
    # Find samples that didn't cycle naturally
    non_cycling_indices = [idx for idx, has_cycle, _, _ in natural_results if not has_cycle]
    print(f"   Natural cycling: {len(texts) - len(non_cycling_indices)}/{len(texts)}")
    print(f"   Non-cycling: {len(non_cycling_indices)}/{len(texts)}")
    
    # Run ICL generation only for non-cycling samples
    print("\n2. Running ICL generation for non-cycling samples...")
    if non_cycling_indices:
        non_cycling_texts = [texts[i] for i in non_cycling_indices]
        icl_results = run_generation(
            model, tokenizer, non_cycling_texts, max_length, max_new_tokens, batch_size, device, use_icl=True
        )
        # Re-map indices
        icl_results = [(non_cycling_indices[i], has_cycle, info, output) 
                       for i, (_, has_cycle, info, output) in enumerate(icl_results)]
    else:
        icl_results = []
    
    icl_cycling_count = sum(1 for _, has_cycle, _, _ in icl_results if has_cycle)
    print(f"   ICL-induced cycling: {icl_cycling_count}/{len(non_cycling_indices)}")
    
    # Categorize all samples
    categories = categorize_samples(natural_results, icl_results)
    
    counts = {
        'natural': sum(1 for c in categories.values() if c == 'natural'),
        'icl_induced': sum(1 for c in categories.values() if c == 'icl_induced'),
        'never': sum(1 for c in categories.values() if c == 'never')
    }
    print(f"\n3. Final categorization:")
    print(f"   Natural: {counts['natural']}")
    print(f"   ICL-induced: {counts['icl_induced']}")
    print(f"   Never: {counts['never']}")
    
    # Compute entropy for each category
    print("\n4. Computing entropy...")
    category_entropies = {'natural': [], 'icl_induced': [], 'never': []}
    
    # Build sequences for entropy computation
    # For each sample, we compute entropy at the cycle start position (or end if no cycle)
    for idx, has_cycle, cycle_info, output_tokens in natural_results:
        cat = categories[idx]
        
        if has_cycle and cycle_info[3] is not None:
            # Use sequence up to cycle start
            cycle_start = cycle_info[3]
            seq = output_tokens[:cycle_start + 1]
        else:
            # Use full sequence
            seq = output_tokens
        
        if len(seq) > 0:
            category_entropies[cat].append(seq)
    
    # Compute actual entropy values
    entropy_values = {}
    for cat, sequences in category_entropies.items():
        if sequences:
            entropy_values[cat] = compute_entropy_for_sequences(
                model, tokenizer, sequences, device, batch_size
            )
        else:
            entropy_values[cat] = []
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return {
        'categories': categories,
        'entropies': entropy_values,
        'counts': counts
    }


def plot_entropy_by_category(results_by_checkpoint, model_name, output_path):
    """
    Plot entropy distribution by category across checkpoints.
    """
    # Build DataFrame
    rows = []
    for checkpoint, result in results_by_checkpoint.items():
        for cat, entropies in result['entropies'].items():
            for ent in entropies:
                rows.append({
                    'checkpoint': checkpoint,
                    'category': cat,
                    'entropy': ent
                })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Checkpoint ordering
    def sort_key(cp):
        if cp == 'steplatest':
            return float('inf')
        return int(cp.replace('step', ''))
    
    checkpoints = sorted(df['checkpoint'].unique(), key=sort_key)
    
    # Nice checkpoint labels
    checkpoint_labels = {}
    for cp in checkpoints:
        label = cp.replace('step', '')
        if label.isdigit() and int(label) >= 1000:
            label = f"{int(label) // 1000}K"
        checkpoint_labels[cp] = label
    
    df['checkpoint_label'] = df['checkpoint'].map(checkpoint_labels)
    df['checkpoint_order'] = df['checkpoint'].map(lambda x: sort_key(x))
    
    # Colors
    palette = {
        'natural': '#2ecc71',      # Green - natural repeating
        'icl_induced': '#3498db',  # Blue - ICL induced
        'never': '#e74c3c'         # Red - never repeating
    }
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Box plot
    ax1 = axes[0]
    order = sorted(df['checkpoint_label'].unique(), key=lambda x: sort_key(x.replace('K', '000') if 'K' in x else ('999999' if x == 'latest' else x)))
    
    sns.boxplot(data=df, x='checkpoint_label', y='entropy', hue='category',
                palette=palette, ax=ax1, order=order)
    ax1.set_xlabel('Checkpoint', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.set_title('Entropy Distribution by Category', fontsize=14)
    ax1.legend(title='Category', loc='upper right')
    
    # Plot 2: Mean entropy evolution
    ax2 = axes[1]
    mean_df = df.groupby(['checkpoint', 'category'])['entropy'].mean().reset_index()
    mean_df['checkpoint_order'] = mean_df['checkpoint'].map(lambda x: sort_key(x))
    mean_df = mean_df.sort_values('checkpoint_order')
    
    for cat in ['natural', 'icl_induced', 'never']:
        cat_df = mean_df[mean_df['category'] == cat]
        if not cat_df.empty:
            ax2.plot(range(len(cat_df)), cat_df['entropy'].values, 
                    marker='o', linewidth=2, markersize=8,
                    color=palette[cat], label=cat.replace('_', ' ').title())
    
    ax2.set_xticks(range(len(checkpoints)))
    ax2.set_xticklabels([checkpoint_labels[cp] for cp in checkpoints])
    ax2.set_xlabel('Checkpoint', fontsize=12)
    ax2.set_ylabel('Mean Entropy', fontsize=12)
    ax2.set_title('Mean Entropy Evolution', fontsize=14)
    ax2.legend(title='Category')
    ax2.grid(True, alpha=0.3)
    
    # Add counts table
    model_short = model_name.split('/')[-1]
    count_text = f"{model_short} Sample Counts:\n"
    for cp in checkpoints:
        if cp in results_by_checkpoint:
            counts = results_by_checkpoint[cp]['counts']
            count_text += f"{checkpoint_labels[cp]}: N={counts['natural']}, I={counts['icl_induced']}, X={counts['never']}\n"
    
    fig.suptitle(f'{model_short}: Entropy Analysis with 3-Way Categorization\n'
                 f'(Natural / ICL-induced / Never repeating)',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Also save CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def main(
    model_name: str = "EleutherAI/pythia-70m",
    n_samples: int = 500,
    batch_size: int = 8,
    max_length: int = 32,
    max_new_tokens: int = 1000,
    seed: int = 42,
    output_dir: str = "/home/mmahaut/projects/parrots/outputs_entropy_3way",
    checkpoints: str = "step1,step1000,step5000,step10000,step100000,steplatest"
):
    """
    Run entropy analysis with proper 3-way categorization across checkpoints.
    """
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load texts
    texts = load_text_dataset(seed=seed, n_samples=n_samples)
    print(f"Loaded {len(texts)} texts")
    
    checkpoint_list = [cp.strip() for cp in checkpoints.split(',')]
    
    results_by_checkpoint = {}
    
    for checkpoint in checkpoint_list:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {checkpoint}")
        print(f"{'='*60}")
        
        revision = checkpoint if checkpoint != 'steplatest' else None
        
        result = run_entropy_analysis(
            model_name=model_name,
            revision=revision,
            texts=texts,
            n_cycles=2,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            device=device
        )
        
        results_by_checkpoint[checkpoint] = result
        
        # Save intermediate results
        model_short = model_name.split('/')[-1]
        cp_csv = output_path / f"entropy_3way_{model_short}_{checkpoint}.csv"
        
        rows = []
        for cat, entropies in result['entropies'].items():
            for ent in entropies:
                rows.append({'category': cat, 'entropy': ent})
        pd.DataFrame(rows).to_csv(cp_csv, index=False)
        print(f"Saved: {cp_csv}")
    
    # Generate final plot
    model_short = model_name.split('/')[-1]
    plot_path = output_path / f"entropy_3way_evolution_{model_short}.png"
    plot_entropy_by_category(results_by_checkpoint, model_name, str(plot_path))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for cp in checkpoint_list:
        if cp in results_by_checkpoint:
            counts = results_by_checkpoint[cp]['counts']
            print(f"{cp}: Natural={counts['natural']}, ICL={counts['icl_induced']}, Never={counts['never']}")


if __name__ == "__main__":
    typer.run(main)
