#!/usr/bin/env python3
"""
Cycle Evolution: Original vs Learnt Repetition
Tracks sequences that were repeating from step1 vs those that learned to repeat later
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from parrots.cycle_detection import detect_cycles

def load_model_checkpoint(model_name, revision=None):
    """Load model at specific checkpoint"""
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

def check_repetition_for_texts(model, tokenizer, texts, max_length=32, max_new_tokens=200, batch_size=8):
    """Check which texts produce repetitive outputs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    tokenizer.padding_side = "left"
    
    repetition_status = {}  # text_idx -> bool
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Checking repetition"):
        batch_texts = texts[i:i+batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(texts))))
        
        toked = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                         truncation=True, max_length=max_length).to(device)
        
        prompt_lengths = toked["attention_mask"].sum(dim=1).tolist()
        
        with torch.no_grad():
            outputs = model.generate(
                **toked, 
                do_sample=False, 
                max_new_tokens=max_new_tokens
            )
        
        for j, idx in enumerate(batch_indices):
            prompt_len = int(prompt_lengths[j])
            gen_ids = outputs[j, prompt_len:].cpu().tolist()
            
            # detect_cycles returns (cycle, cycle_size, cycle_count)
            cycle, cycle_size, cycle_count = detect_cycles(gen_ids)
            is_repeating = cycle is not None and cycle_size > 0 and cycle_count > 1
            repetition_status[idx] = is_repeating
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return repetition_status

def load_dataset(n_samples=500, seed=42):
    """Load text dataset for analysis"""
    try:
        sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src')
        from modules.cached_data_utils import load_text_dataset
        return load_text_dataset(n_samples=n_samples, seed=seed)
    except:
        from datasets import load_dataset
        np.random.seed(seed)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds['text'] if len(t.split()) > 10][:n_samples * 2]
        np.random.shuffle(texts)
        return texts[:n_samples]

def categorize_repetition_evolution(all_checkpoint_status, checkpoints, n_texts):
    """
    Categorize each text by when it first started repeating:
    - 'never': never repeats at any checkpoint
    - 'original': repeating from step1 (originally repeating)
    - 'learnt_early': started repeating at step1000 or step5000
    - 'learnt_late': started repeating at step10000 or later
    """
    categories = {}
    
    for idx in range(n_texts):
        first_repeating = None
        
        for cp in checkpoints:
            if cp in all_checkpoint_status and idx in all_checkpoint_status[cp]:
                if all_checkpoint_status[cp][idx]:
                    first_repeating = cp
                    break
        
        if first_repeating is None:
            categories[idx] = 'never'
        elif first_repeating == 'step1':
            categories[idx] = 'original'
        elif first_repeating in ['step1000', 'step5000']:
            categories[idx] = 'learnt_early'
        else:
            categories[idx] = 'learnt_late'
    
    return categories

def create_evolution_plot(all_checkpoint_status, checkpoints, model_name, output_dir):
    """Create cycle evolution plot showing original vs learnt repetition"""
    
    # Count per category at each checkpoint
    n_texts = max(max(all_checkpoint_status[cp].keys()) for cp in checkpoints if cp in all_checkpoint_status) + 1
    
    categories = categorize_repetition_evolution(all_checkpoint_status, checkpoints, n_texts)
    
    # Track repetition counts by category at each checkpoint
    category_counts = {cp: {'original': 0, 'learnt_early': 0, 'learnt_late': 0, 'never': 0} 
                       for cp in checkpoints}
    
    for idx, cat in categories.items():
        for cp in checkpoints:
            if cp in all_checkpoint_status and idx in all_checkpoint_status[cp]:
                if all_checkpoint_status[cp][idx]:  # Is repeating at this checkpoint
                    category_counts[cp][cat] += 1
    
    # Create stacked bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stacked bars showing composition of repetitive sequences
    x = np.arange(len(checkpoints))
    width = 0.6
    
    original = [category_counts[cp]['original'] for cp in checkpoints]
    learnt_early = [category_counts[cp]['learnt_early'] for cp in checkpoints]
    learnt_late = [category_counts[cp]['learnt_late'] for cp in checkpoints]
    
    ax1.bar(x, original, width, label='Original (since step1)', color='#e74c3c', alpha=0.85)
    ax1.bar(x, learnt_early, width, bottom=original, label='Learnt Early (step1000-5000)', color='#f39c12', alpha=0.85)
    ax1.bar(x, learnt_late, width, bottom=np.array(original)+np.array(learnt_early), 
            label='Learnt Late (step10000+)', color='#9b59b6', alpha=0.85)
    
    ax1.set_xlabel('Training Checkpoint', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Repetitive Sequences', fontsize=12, fontweight='bold')
    ax1.set_title(f'Repetition Composition by Origin\n{model_name}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(checkpoints, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Line plot showing total and breakdown
    total_repeating = [sum(category_counts[cp].values()) - category_counts[cp]['never'] for cp in checkpoints]
    
    ax2.plot(x, total_repeating, 'ko-', label='Total Repeating', linewidth=2, markersize=10)
    ax2.plot(x, original, 'r--', label='Original', linewidth=2, marker='o', markersize=6)
    ax2.fill_between(x, 0, learnt_early, alpha=0.3, color='#f39c12', label='Learnt Early')
    ax2.fill_between(x, 0, learnt_late, alpha=0.3, color='#9b59b6', label='Learnt Late')
    
    ax2.set_xlabel('Training Checkpoint', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax2.set_title(f'Original vs Learnt Repetition\n{model_name}', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(checkpoints, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_model = model_name.replace("/", "_")
    plot_path = output_dir / f"cycle_evolution_original_vs_learnt_{safe_model}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to {plot_path}")
    plt.close()
    
    # Print summary
    print("\n📊 Category Summary:")
    print(f"   Original (step1): {sum(1 for c in categories.values() if c == 'original')}")
    print(f"   Learnt Early: {sum(1 for c in categories.values() if c == 'learnt_early')}")
    print(f"   Learnt Late: {sum(1 for c in categories.values() if c == 'learnt_late')}")
    print(f"   Never Repeats: {sum(1 for c in categories.values() if c == 'never')}")

def main():
    parser = argparse.ArgumentParser(description="Cycle evolution: original vs learnt repetition")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--checkpoints", type=str, nargs="+", 
                       default=["step1", "step1000", "step5000", "step10000", "step100000", "steplatest"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./cycle_evolution_results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Cycle Evolution: Original vs Learnt Repetition")
    print(f"📊 Model: {args.model_name}")
    print(f"📊 Checkpoints: {args.checkpoints}")
    
    # Load dataset ONCE (same texts across all checkpoints)
    print("\n📚 Loading dataset...")
    texts = load_dataset(args.n_samples, args.seed)
    print(f"   Loaded {len(texts)} texts")
    
    # Track repetition status at each checkpoint
    all_checkpoint_status = {}
    
    for checkpoint in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"📈 Analyzing checkpoint: {checkpoint}")
        print('='*60)
        
        try:
            model, tokenizer = load_model_checkpoint(args.model_name, checkpoint)
            
            status = check_repetition_for_texts(
                model, tokenizer, texts,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size
            )
            
            all_checkpoint_status[checkpoint] = status
            
            n_repeating = sum(status.values())
            print(f"   Repeating: {n_repeating}/{len(texts)} ({100*n_repeating/len(texts):.1f}%)")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save results
    import json
    safe_model = args.model_name.replace("/", "_")
    results_file = output_dir / f"cycle_evolution_status_{safe_model}.json"
    
    # Convert to serializable format
    serializable = {cp: {str(k): v for k, v in status.items()} 
                    for cp, status in all_checkpoint_status.items()}
    with open(results_file, 'w') as f:
        json.dump(serializable, f)
    print(f"\n💾 Status saved to {results_file}")
    
    # Create visualization
    create_evolution_plot(all_checkpoint_status, args.checkpoints, args.model_name, output_dir)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
