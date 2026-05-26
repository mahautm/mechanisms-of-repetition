#!/usr/bin/env python3
"""
Script to compute entropy of output logit distributions for no-cycle-ICL and natural data
across different cycles (0-4) using the same data pipeline as contrast_analysis.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu')

from modules.model_utils import HookedModel, load_model_and_tokenizer, get_device
from modules.data_utils import load_text_dataset
from modules.contrast_analysis import extract_contrasts
from torch.nn.functional import softmax, log_softmax
from torch.amp import autocast
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from parrots.cycle_detection import detect_cycles
import seaborn as sns

def compute_entropy(logits, temperature=1.0):
    """
    Compute entropy of probability distribution from logits
    
    Args:
        logits: torch.Tensor of shape (batch_size, vocab_size) or (vocab_size,)
        temperature: Temperature for softmax (default 1.0)
    
    Returns:
        entropy: torch.Tensor of entropies
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Compute log probabilities (numerically stable)
    log_probs = log_softmax(scaled_logits, dim=-1)
    
    # Compute probabilities
    probs = softmax(scaled_logits, dim=-1)
    
    # Compute entropy: H = -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy

def extract_entropy_from_data(text, hooked_model, tokenizer, n_cycles=0, batch_size=4, max_length=64, max_new_tokens=20):
    """
    Extract entropy data using the same pipeline as contrast analysis but computing entropy instead of contrasts.
    This follows the exact same data separation logic as extract_contrasts.
    Ultra memory-optimized version with very small batches and sequences.
    """
    natural_entropies = []
    icl_entropies = []
    no_cycle_icl_entropies = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = min(batch_size, len(text))
    
    # Set padding side for decoder-only models
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Pre-tokenize all text in advance for speed
    pretokenized = [tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length) for t in text]
    
    for i in tqdm(range(0, len(text), batch_size), desc="Analyzing samples for entropy", total=(len(text) + batch_size - 1)//batch_size):
        batch = pretokenized[i:i+batch_size]
        
        # Pad input_ids and attention_mask to the same length before concatenation
        input_ids_list = [b['input_ids'].squeeze(0) for b in batch]
        attention_mask_list = [b['attention_mask'].squeeze(0) for b in batch]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        # Ensure correct data types: input_ids should be long, attention_mask should be long
        toked = {
            'input_ids': input_ids.long().to(device), 
            'attention_mask': attention_mask.long().to(device)
        }
        
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.float):
            o1 = hooked_model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens)
        
        # Clear GPU cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Find prompt lengths
        plengths = toked["attention_mask"].sum(dim=1).tolist()
        
        # Ensure o1 and all tensors are on CPU for detect_cycles
        o1_cpu = [o1[j].detach().cpu() if o1[j].device.type != 'cpu' else o1[j] for j in range(len(o1))]
        # Remove all padding tokens from generated output for cycle detection
        o1_cpu = [o[o != tokenizer.pad_token_id] for o in o1_cpu]
        
        reps = []
        for dbg_idx, o in enumerate(tqdm(o1_cpu, desc=f"cycle_detect batch {i}", leave=False)):
            rep = detect_cycles(o[plengths[dbg_idx]:], return_index=True, pad_token_id=tokenizer.pad_token_id)
            reps.append(rep)
        
        # Build natural and ICL inputs (same logic as contrast_analysis)
        # Ensure all token IDs are integers and handle n_cycles=0 case
        natural_input = []
        icl_input = []
        no_cycle_icl_input = []
        
        for j, rep in enumerate(reps):
            if rep[0] is not None:
                # Has cycles
                base_natural = o1_cpu[j][:plengths[j] + rep[3]].tolist()
                if n_cycles > 0:
                    # Add repeated cycles
                    cycle_tokens = rep[0] * n_cycles
                    natural_seq = [int(token) for token in (base_natural + cycle_tokens)]
                    icl_seq = [int(token) for token in cycle_tokens]
                else:
                    # n_cycles = 0, use base sequence only
                    natural_seq = [int(token) for token in base_natural]
                    icl_seq = []  # Empty for cycle 0
                
                if len(natural_seq) > 0:
                    natural_input.append(natural_seq)
                if len(icl_seq) > 0:
                    icl_input.append(icl_seq)
            else:
                # No cycles detected
                base_no_cycle = o1_cpu[j][:plengths[j]].tolist()
                if n_cycles > 0:
                    # Repeat the base sequence
                    no_cycle_seq = [int(token) for token in (base_no_cycle * n_cycles)]
                else:
                    # n_cycles = 0, use base sequence only
                    no_cycle_seq = [int(token) for token in base_no_cycle]
                
                if len(no_cycle_seq) > 0:
                    no_cycle_icl_input.append(no_cycle_seq)
        
        print(f"Batch {i}, cycle {n_cycles}: natural={len(natural_input)}, icl={len(icl_input)}, no_cycle={len(no_cycle_icl_input)}")
        print(f"  Sample lengths - natural: {[len(s) for s in natural_input[:3]]}, icl: {[len(s) for s in icl_input[:3]]}, no_cycle: {[len(s) for s in no_cycle_icl_input[:3]]}")
        
        # Helper for padding and batching
        def pad_and_batch(seqs):
            if not seqs:
                return None, None
            
            # Filter out empty sequences
            non_empty_seqs = [seq for seq in seqs if len(seq) > 0]
            if not non_empty_seqs:
                return None, None
                
            max_len = max(len(seq) for seq in non_empty_seqs)
            if max_len == 0:
                return None, None
                
            input_ids = [([tokenizer.pad_token_id] * (max_len - len(seq)) + seq) for seq in non_empty_seqs]
            attention_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in non_empty_seqs]
            # Ensure input_ids are long integers (not float)
            return torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(attention_mask, dtype=torch.long, device=device)
        
        # Process natural data
        if natural_input:
            natural_input_ids, natural_attention_mask = pad_and_batch(natural_input)
            if natural_input_ids is not None and natural_input_ids.shape[0] > 0:
                natural_dict = {'input_ids': natural_input_ids, 'attention_mask': natural_attention_mask}
                
                print(f"Natural data - batch size: {natural_input_ids.shape[0]}, seq_len: {natural_input_ids.shape[1]}")
                
                with torch.no_grad():
                    natural_outputs = hooked_model.model(**natural_dict)
                    natural_logits = natural_outputs.logits[:, -1, :].float()  # Convert to float32
                    natural_entropy_batch = compute_entropy(natural_logits)
                    natural_entropies.extend(natural_entropy_batch.cpu().numpy().tolist())
            else:
                print(f"Skipping natural data - empty or invalid sequences")
        
        # Process ICL data
        if icl_input and any(len(seq) > 0 for seq in icl_input):
            non_empty_icl_input = [seq for seq in icl_input if len(seq) > 0]
            if non_empty_icl_input:
                icl_input_ids, icl_attention_mask = pad_and_batch(non_empty_icl_input)
                if icl_input_ids is not None and icl_input_ids.shape[0] > 0:
                    icl_dict = {'input_ids': icl_input_ids, 'attention_mask': icl_attention_mask}
                    
                    print(f"ICL data - batch size: {icl_input_ids.shape[0]}, seq_len: {icl_input_ids.shape[1]}")
                    
                    with torch.no_grad():
                        icl_outputs = hooked_model.model(**icl_dict)
                        icl_logits = icl_outputs.logits[:, -1, :].float()  # Convert to float32
                        icl_entropy_batch = compute_entropy(icl_logits)
                        icl_entropies.extend(icl_entropy_batch.cpu().numpy().tolist())
                else:
                    print(f"Skipping ICL data - empty or invalid sequences")
        
        # Process no-cycle ICL data
        if no_cycle_icl_input:
            no_cycle_input_ids, no_cycle_attention_mask = pad_and_batch(no_cycle_icl_input)
            if no_cycle_input_ids is not None and no_cycle_input_ids.shape[0] > 0:
                no_cycle_dict = {'input_ids': no_cycle_input_ids, 'attention_mask': no_cycle_attention_mask}
                
                print(f"No-cycle ICL data - batch size: {no_cycle_input_ids.shape[0]}, seq_len: {no_cycle_input_ids.shape[1]}")
                
                with torch.no_grad():
                    no_cycle_outputs = hooked_model.model(**no_cycle_dict)
                    no_cycle_logits = no_cycle_outputs.logits[:, -1, :].float()  # Convert to float32
                    no_cycle_entropy_batch = compute_entropy(no_cycle_logits)
                    no_cycle_icl_entropies.extend(no_cycle_entropy_batch.cpu().numpy().tolist())
            else:
                print(f"Skipping no-cycle ICL data - empty or invalid sequences")
        
        # Clear memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear hooked model outputs
        hooked_model.clear()
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    return {
        'natural': np.array(natural_entropies),
        'icl': np.array(icl_entropies), 
        'no_cycle_icl': np.array(no_cycle_icl_entropies)
    }

def analyze_entropy_evolution(model_name="EleutherAI/pythia-1.4b", revision=None, max_samples=200, batch_size=4, device="cuda"):
    """
    Analyze entropy evolution across cycles for different data types using the same data pipeline
    Memory-optimized version with very small batches and aggressive memory clearing
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer using the project's utility functions
    model, tokenizer = load_model_and_tokenizer(model_name, revision, use_bfloat16=True)  # Use bfloat16 to save memory
    model.eval()
    device = get_device()
    model.to(device)
    
    # Create hooked model (no specific hook layer needed for this analysis)
    hooked_model = HookedModel(model, layer=None)
    
    print(f"Loaded model {model_name}")
    
    # Use much smaller dataset for memory efficiency
    texts = load_text_dataset(seed=42, n_samples=max_samples)
    print(f"Loaded {len(texts)} text samples")
    
    results = {
        'cycle': [],
        'data_type': [],
        'entropy_values': []
    }
    
    cycles = list(range(5))  # 0 to 4
    
    print("Computing entropies across cycles...")
    
    for cycle in cycles:
        print(f"Processing cycle {cycle}...")
        
        # Clear all caches before each cycle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract entropy data for this cycle using the same pipeline as contrast analysis
        entropy_data = extract_entropy_from_data(
            texts, hooked_model, tokenizer, 
            n_cycles=cycle, batch_size=4,  # Very small batch size
            max_length=64, max_new_tokens=20  # Much shorter sequences
        )
        
        # Store results for each data type
        for data_type, entropies in entropy_data.items():
            if len(entropies) > 0:
                results['cycle'].extend([cycle] * len(entropies))
                results['data_type'].extend([data_type] * len(entropies))
                results['entropy_values'].extend(entropies.tolist())
                print(f"  {data_type}: {len(entropies)} samples, mean entropy: {entropies.mean():.4f}")
        
        # Clear GPU cache between cycles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force garbage collection
        import gc
        gc.collect()
    
    return pd.DataFrame(results)

def plot_entropy_evolution(df, save_path=None):
    """
    Create plots showing entropy evolution across cycles (excluding ICL data)
    """
    # Filter out ICL data
    df_filtered = df[df['data_type'] != 'icl'].copy()
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean entropy evolution
    summary_df = df_filtered.groupby(['cycle', 'data_type'])['entropy_values'].agg(['mean', 'std']).reset_index()
    summary_df.columns = ['cycle', 'data_type', 'entropy_mean', 'entropy_std']
    
    for data_type in ['natural', 'no_cycle_icl']:
        data_subset = summary_df[summary_df['data_type'] == data_type]
        if not data_subset.empty:
            ax1.plot(data_subset['cycle'], data_subset['entropy_mean'], 
                    marker='o', linewidth=2, markersize=8, label=data_type.replace('_', ' ').title())
            ax1.fill_between(data_subset['cycle'], 
                            data_subset['entropy_mean'] - data_subset['entropy_std'],
                            data_subset['entropy_mean'] + data_subset['entropy_std'],
                            alpha=0.3)
    
    ax1.set_xlabel('Cycle Number', fontsize=12)
    ax1.set_ylabel('Mean Entropy (nats)', fontsize=12)
    ax1.set_title('Entropy Evolution Across Cycles (Natural vs No-Cycle ICL)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(5))
    
    # Plot 2: Entropy distributions by cycle
    sns.boxplot(data=df_filtered, x='cycle', y='entropy_values', hue='data_type', ax=ax2)
    ax2.set_xlabel('Cycle Number', fontsize=12)
    ax2.set_ylabel('Entropy (nats)', fontsize=12)
    ax2.set_title('Entropy Distributions by Cycle (Natural vs No-Cycle ICL)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved entropy evolution plot to {save_path}")
    
    plt.show()

def main():
    """
    Main function to run entropy analysis using the same data pipeline as contrast analysis
    """
    print("Starting entropy analysis of output logit distributions...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Run entropy analysis with increased sample size
        df = analyze_entropy_evolution(
            model_name="EleutherAI/pythia-1.4b",
            revision=None,  # steplatest equivalent
            max_samples=1000,  # Increased sample size
            batch_size=4,  # Keep small batch size for memory management
            device=device
        )
        
        # Print summary statistics
        print("\nSummary Statistics:")
        summary = df.groupby(['cycle', 'data_type'])['entropy_values'].agg(['count', 'mean', 'std']).round(4)
        print(summary)
        
        # Create plots
        plot_entropy_evolution(df, save_path="/home/mmahaut/projects/parrots/logit_entropy_evolution.png")
        
        # Save detailed results
        df.to_csv("/home/mmahaut/projects/parrots/logit_entropy_results.csv", index=False)
        print("Saved detailed results to logit_entropy_results.csv")
        
    except Exception as e:
        print(f"Error during entropy analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()