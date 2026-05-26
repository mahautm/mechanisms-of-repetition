#!/usr/bin/env python3
"""
Multihead attention analysis split by Original vs Acquired repetition
Uses the same trained lenses but categorizes sequences based on when they first started repeating

Original: sequences repeating at step1 (always had the behavior)
Acquired: sequences that first started repeating at a later checkpoint
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import json
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from parrots.cycle_detection import detect_cycles
from parrots.aa_fortu.modules.data_utils import load_text_dataset
from parrots.aa_fortu.aa_fortu_train_multihead_lens import MultiHeadLens
from parrots.aa_fortu.modules.model_utils import HookedModel
import re

def load_cycle_evolution_categories(evolution_file, checkpoints):
    """
    Load cycle evolution data and categorize sequences.
    
    Returns:
        dict mapping idx -> 'original' | 'acquired' | 'never'
    """
    with open(evolution_file, 'r') as f:
        data = json.load(f)
    
    # Convert to proper format
    all_status = {cp: {int(k): v for k, v in status.items()} 
                  for cp, status in data.items()}
    
    n_texts = max(max(status.keys()) for status in all_status.values()) + 1
    
    categories = {}
    first_checkpoint = checkpoints[0]  # step1
    
    for idx in range(n_texts):
        first_rep_cp = None
        for cp in checkpoints:
            if cp in all_status and idx in all_status[cp]:
                if all_status[cp][idx]:  # is repeating
                    first_rep_cp = cp
                    break
        
        if first_rep_cp is None:
            categories[idx] = 'never'
        elif first_rep_cp == first_checkpoint:
            categories[idx] = 'original'
        else:
            categories[idx] = 'acquired'
    
    return categories

def compute_head_contrast(model, tokenizer, lens, text, layer_idx, device, max_new_tokens=100):
    """
    Generate text and compute attention head contrasts at cycle start position.
    
    Returns:
        dict: {head_idx: contrast_value} or None if no cycle
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32).to(device)
    input_len = inputs['input_ids'].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            output_attentions=False,
            return_dict_in_generate=True
        )
    
    generated_ids = outputs.sequences[0].tolist()
    
    # Detect cycle
    cycle, cycle_size, cycle_count = detect_cycles(generated_ids)
    
    if cycle_count == 0:
        return None, False
    
    # Find cycle start position (relative to generation start)
    # The cycle detection gives the cycle in the full sequence
    # We need to find where it starts
    cycle_start = None
    for i in range(input_len, len(generated_ids) - cycle_size + 1):
        if generated_ids[i:i+cycle_size] == cycle:
            cycle_start = i - input_len  # Relative to generation
            break
    
    if cycle_start is None:
        return None, True
    
    # TODO: Hook attention at cycle_start and compute contrasts
    # For now, return placeholder - full implementation needs HookedModel
    return {'cycle_start': cycle_start, 'cycle_size': cycle_size}, True

def run_analysis_with_categories(
    model_name: str,
    revision: str,
    lens_path: str,
    layer_idx: int,
    categories: dict,
    n_samples: int = 300,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Run multihead analysis split by original vs acquired categories.
    """
    print(f"\nLoading model {model_name} (revision: {revision or 'latest'})...")
    
    # Load model
    if revision and revision != 'steplatest':
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
    
    # Load lens
    lens_files = list(Path(lens_path).glob(f"layer_{layer_idx}*.pth"))
    if not lens_files:
        raise FileNotFoundError(f"No lens found for layer {layer_idx} in {lens_path}")
    
    lens_data = torch.load(lens_files[0], weights_only=False, map_location=device)
    if isinstance(lens_data, dict):
        lens = MultiHeadLens.from_dict(lens_data)
    else:
        lens = lens_data
    lens.to(device)
    print(f"Loaded lens: {lens_files[0]}")
    
    # Load texts
    texts = load_text_dataset(seed=seed, n_samples=n_samples)
    
    # Split by category
    original_indices = [i for i, cat in categories.items() if cat == 'original' and i < len(texts)]
    acquired_indices = [i for i, cat in categories.items() if cat == 'acquired' and i < len(texts)]
    
    print(f"\nCategory split at {revision or 'steplatest'}:")
    print(f"  Original: {len(original_indices)} sequences")
    print(f"  Acquired: {len(acquired_indices)} sequences")
    
    results = {
        'original': {'repeating': 0, 'total': len(original_indices), 'contrasts': []},
        'acquired': {'repeating': 0, 'total': len(acquired_indices), 'contrasts': []}
    }
    
    # Process original sequences
    print("\nProcessing original sequences...")
    for idx in tqdm(original_indices[:100]):  # Limit for speed
        contrast, is_repeating = compute_head_contrast(
            model, tokenizer, lens, texts[idx], layer_idx, device
        )
        if is_repeating:
            results['original']['repeating'] += 1
        if contrast:
            results['original']['contrasts'].append(contrast)
    
    # Process acquired sequences
    print("\nProcessing acquired sequences...")
    for idx in tqdm(acquired_indices[:100]):  # Limit for speed
        contrast, is_repeating = compute_head_contrast(
            model, tokenizer, lens, texts[idx], layer_idx, device
        )
        if is_repeating:
            results['acquired']['repeating'] += 1
        if contrast:
            results['acquired']['contrasts'].append(contrast)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--lens-path", type=str, required=True)
    parser.add_argument("--evolution-file", type=str, required=True,
                       help="JSON file from cycle_evolution analysis")
    parser.add_argument("--layer-idx", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="./outputs_original_vs_acquired")
    
    args = parser.parse_args()
    
    checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
    
    # Load categories
    print("Loading sequence categories from cycle evolution data...")
    categories = load_cycle_evolution_categories(args.evolution_file, checkpoints)
    
    n_original = sum(1 for c in categories.values() if c == 'original')
    n_acquired = sum(1 for c in categories.values() if c == 'acquired')
    n_never = sum(1 for c in categories.values() if c == 'never')
    
    print(f"Categories: {n_original} original, {n_acquired} acquired, {n_never} never")
    
    # Run analysis
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = run_analysis_with_categories(
        model_name=args.model_name,
        revision=args.revision,
        lens_path=args.lens_path,
        layer_idx=args.layer_idx,
        categories=categories,
        n_samples=args.n_samples,
        device=device
    )
    
    # Summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for cat in ['original', 'acquired']:
        r = results[cat]
        rep_rate = 100 * r['repeating'] / max(1, min(100, r['total']))
        print(f"\n{cat.upper()}:")
        print(f"  Processed: {min(100, r['total'])}")
        print(f"  Repeating: {r['repeating']} ({rep_rate:.1f}%)")
        print(f"  Contrasts captured: {len(r['contrasts'])}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = args.revision or 'steplatest'
    output_file = output_dir / f"original_vs_acquired_{args.model_name.replace('/', '_')}_{checkpoint_name}_L{args.layer_idx}.json"
    
    # Convert to serializable format
    save_results = {
        'model': args.model_name,
        'checkpoint': checkpoint_name,
        'layer': args.layer_idx,
        'original': {
            'total': results['original']['total'],
            'repeating': results['original']['repeating'],
            'n_contrasts': len(results['original']['contrasts'])
        },
        'acquired': {
            'total': results['acquired']['total'],
            'repeating': results['acquired']['repeating'],
            'n_contrasts': len(results['acquired']['contrasts'])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
