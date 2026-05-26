#!/usr/bin/env python3
"""
Entropy Evolution Analysis across Training Checkpoints
Measures how output entropy evolves during training for Natural, ICL, and No-Cycle-ICL sequences
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.nn.functional import softmax, log_softmax
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from parrots.cycle_detection import detect_cycles

def compute_entropy(logits, temperature=1.0):
    """Compute entropy of probability distribution from logits"""
    scaled_logits = logits / temperature
    log_probs = log_softmax(scaled_logits, dim=-1)
    probs = softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def load_model_checkpoint(model_name, revision=None):
    """Load model at specific checkpoint"""
    print(f"Loading {model_name}" + (f" at {revision}" if revision else ""))
    
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    
    if revision and revision != "steplatest":
        kwargs["revision"] = revision
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        revision=revision if revision and revision != "steplatest" else None,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def analyze_entropy_for_checkpoint(model, tokenizer, texts, max_length=32, max_new_tokens=100, batch_size=8, n_cycles=4):
    """
    Analyze entropy for Natural, ICL, and No-Cycle-ICL conditions at a checkpoint.
    Tracks entropy at each cycle position.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    tokenizer.padding_side = "left"
    
    results = {
        'natural': {'entropies': [], 'cycle_entropies': {i: [] for i in range(n_cycles + 1)}},
        'icl': {'entropies': [], 'cycle_entropies': {i: [] for i in range(n_cycles + 1)}},
        'no_cycle_icl': {'entropies': [], 'cycle_entropies': {i: [] for i in range(n_cycles + 1)}},
    }
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize prompts
        toked = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                         truncation=True, max_length=max_length).to(device)
        
        prompt_lengths = toked["attention_mask"].sum(dim=1).tolist()
        
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            # Generate with output scores
            outputs = model.generate(
                **toked, 
                do_sample=False, 
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        generated_ids = outputs.sequences
        scores = outputs.scores  # List of (batch, vocab) tensors
        
        # Process each sample in batch
        for j in range(len(batch_texts)):
            prompt_len = prompt_lengths[j]
            
            # Get generated portion
            gen_ids = generated_ids[j, prompt_len:].cpu()
            
            # Detect cycles - returns (cycle, cycle_size, cycle_count)
            cycle_result = detect_cycles(gen_ids.tolist())
            cycle, cycle_size, cycle_count = cycle_result
            is_repeating = cycle is not None and cycle_size > 0 and cycle_count > 1
            
            # Compute entropy for each generated token
            sample_entropies = []
            for k, score in enumerate(scores):
                if k < len(scores):
                    ent = compute_entropy(score[j:j+1]).item()
                    sample_entropies.append(ent)
            
            if len(sample_entropies) == 0:
                continue
            
            mean_entropy = np.mean(sample_entropies)
            
            # Classify sequence
            if is_repeating:
                results['natural']['entropies'].append(mean_entropy)
                # Track entropy at cycle boundaries
                if cycle_size > 0:
                    for c in range(min(n_cycles + 1, len(sample_entropies) // max(cycle_size, 1))):
                        start_idx = c * cycle_size
                        end_idx = min((c + 1) * cycle_size, len(sample_entropies))
                        if start_idx < len(sample_entropies):
                            cycle_ent = np.mean(sample_entropies[start_idx:end_idx])
                            results['natural']['cycle_entropies'][c].append(cycle_ent)
            else:
                results['no_cycle_icl']['entropies'].append(mean_entropy)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def load_dataset(n_samples=500, seed=42):
    """Load text dataset for analysis"""
    try:
        sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src')
        from modules.cached_data_utils import load_text_dataset
        return load_text_dataset(n_samples=n_samples, seed=seed)
    except:
        # Fallback to simple dataset
        from datasets import load_dataset
        np.random.seed(seed)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds['text'] if len(t.split()) > 10][:n_samples * 2]
        np.random.shuffle(texts)
        return texts[:n_samples]

def main():
    parser = argparse.ArgumentParser(description="Entropy evolution across checkpoints")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--checkpoints", type=str, nargs="+", 
                       default=["step1", "step1000", "step5000", "step10000", "step100000", "steplatest"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./entropy_evolution_results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model = args.model_name.replace("/", "_")
    
    print(f"🚀 Entropy Evolution Analysis for {args.model_name}")
    print(f"📊 Checkpoints: {args.checkpoints}")
    
    # Load dataset once
    print("📚 Loading dataset...")
    texts = load_dataset(args.n_samples, args.seed)
    print(f"   Loaded {len(texts)} texts")
    
    # Analyze each checkpoint
    all_results = {}
    
    for checkpoint in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"📈 Analyzing checkpoint: {checkpoint}")
        print('='*60)
        
        try:
            model, tokenizer = load_model_checkpoint(args.model_name, checkpoint)
            
            results = analyze_entropy_for_checkpoint(
                model, tokenizer, texts,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size
            )
            
            all_results[checkpoint] = {
                'natural_mean': np.mean(results['natural']['entropies']) if results['natural']['entropies'] else 0,
                'natural_std': np.std(results['natural']['entropies']) if results['natural']['entropies'] else 0,
                'natural_count': len(results['natural']['entropies']),
                'no_cycle_mean': np.mean(results['no_cycle_icl']['entropies']) if results['no_cycle_icl']['entropies'] else 0,
                'no_cycle_std': np.std(results['no_cycle_icl']['entropies']) if results['no_cycle_icl']['entropies'] else 0,
                'no_cycle_count': len(results['no_cycle_icl']['entropies']),
                'cycle_entropies': {k: np.mean(v) if v else 0 for k, v in results['natural']['cycle_entropies'].items()}
            }
            
            print(f"   Natural (repeating): {all_results[checkpoint]['natural_count']} samples, "
                  f"entropy={all_results[checkpoint]['natural_mean']:.3f}±{all_results[checkpoint]['natural_std']:.3f}")
            print(f"   No-Cycle: {all_results[checkpoint]['no_cycle_count']} samples, "
                  f"entropy={all_results[checkpoint]['no_cycle_mean']:.3f}±{all_results[checkpoint]['no_cycle_std']:.3f}")
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ❌ Error processing {checkpoint}: {e}")
            all_results[checkpoint] = None
    
    # Save results
    results_file = output_dir / f"entropy_evolution_{safe_model}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    
    # Create visualization
    create_entropy_plot(all_results, args.checkpoints, args.model_name, output_dir)
    
    print("\n✅ Analysis complete!")

def create_entropy_plot(all_results, checkpoints, model_name, output_dir):
    """Create entropy evolution plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_checkpoints = [cp for cp in checkpoints if all_results.get(cp) is not None]
    x = range(len(valid_checkpoints))
    
    natural_means = [all_results[cp]['natural_mean'] for cp in valid_checkpoints]
    natural_stds = [all_results[cp]['natural_std'] for cp in valid_checkpoints]
    nocycle_means = [all_results[cp]['no_cycle_mean'] for cp in valid_checkpoints]
    nocycle_stds = [all_results[cp]['no_cycle_std'] for cp in valid_checkpoints]
    
    ax.errorbar(x, natural_means, yerr=natural_stds, marker='o', capsize=5,
                label='Natural (Repetitive)', color='#e74c3c', linewidth=2, markersize=8)
    ax.errorbar(x, nocycle_means, yerr=nocycle_stds, marker='s', capsize=5,
                label='No-Cycle (Non-Repetitive)', color='#3498db', linewidth=2, markersize=8)
    
    ax.set_xlabel('Training Checkpoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Output Entropy', fontsize=12, fontweight='bold')
    ax.set_title(f'Entropy Evolution During Training\n{model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_checkpoints, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_model = model_name.replace("/", "_")
    plot_path = output_dir / f"entropy_evolution_{safe_model}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
