#!/usr/bin/env python3
"""
Full multihead attention analysis with 2x2 categorization:
- Prompt style: Natural vs ICL
- Repetition origin: Original (step1) vs Acquired (later checkpoint)

This captures all combinations:
- Original + Natural
- Original + ICL  
- Acquired + Natural
- Acquired + ICL

Uses same trained lenses, just categorizes data on both dimensions.
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
    Load cycle evolution data and categorize sequences by WHEN they first started repeating.
    
    Returns:
        dict mapping idx -> 'original' | 'acquired' | 'never'
    """
    with open(evolution_file, 'r') as f:
        data = json.load(f)
    
    all_status = {cp: {int(k): v for k, v in status.items()} 
                  for cp, status in data.items()}
    
    n_texts = max(max(status.keys()) for status in all_status.values()) + 1
    
    categories = {}
    first_checkpoint = checkpoints[0]  # step1
    
    for idx in range(n_texts):
        first_rep_cp = None
        for cp in checkpoints:
            if cp in all_status and idx in all_status[cp]:
                if all_status[cp][idx]:
                    first_rep_cp = cp
                    break
        
        if first_rep_cp is None:
            categories[idx] = 'never'
        elif first_rep_cp == first_checkpoint:
            categories[idx] = 'original'
        else:
            categories[idx] = 'acquired'
    
    return categories

def format_icl_prompt(text, n_cycles=4):
    """Create ICL-style prompt with repeated examples"""
    return (text + "\n") * n_cycles + text

def generate_and_analyze(model, tokenizer, text, max_new_tokens=1000, max_length=32, device='cuda'):
    """
    Generate text and return cycle info.
    
    Returns:
        is_repeating: bool
        cycle_info: dict with cycle details or None
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_len = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0].tolist()
    cycle, cycle_size, cycle_count = detect_cycles(generated_ids)
    
    is_repeating = cycle_count > 0
    
    cycle_info = None
    if is_repeating and cycle:
        # Find cycle start position
        cycle_start = None
        for i in range(input_len, len(generated_ids) - cycle_size + 1):
            if generated_ids[i:i+cycle_size] == cycle:
                cycle_start = i - input_len
                break
        
        cycle_info = {
            'cycle_size': cycle_size,
            'cycle_count': cycle_count,
            'cycle_start': cycle_start
        }
    
    return is_repeating, cycle_info

def run_full_2x2_analysis(
    model_name: str,
    revision: str,
    origin_categories: dict,
    texts: list,
    n_cycles: int = 4,
    max_length: int = 32,
    max_new_tokens: int = 1000,
    device: str = 'cuda'
):
    """
    Run analysis capturing both dimensions:
    - Natural vs ICL (prompt style)
    - Original vs Acquired (repetition origin)
    
    Returns results for all 4 combinations.
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
    
    # Initialize 2x2 results structure
    results = {
        'original_natural': {'repeating': 0, 'not_repeating': 0, 'indices': []},
        'original_icl': {'repeating': 0, 'not_repeating': 0, 'indices': []},
        'acquired_natural': {'repeating': 0, 'not_repeating': 0, 'indices': []},
        'acquired_icl': {'repeating': 0, 'not_repeating': 0, 'indices': []},
        'never_natural': {'repeating': 0, 'not_repeating': 0, 'indices': []},
        'never_icl': {'repeating': 0, 'not_repeating': 0, 'indices': []},
    }
    
    print(f"\nProcessing {len(texts)} texts...")
    
    for idx, text in enumerate(tqdm(texts)):
        if idx not in origin_categories:
            continue
        
        origin = origin_categories[idx]  # 'original', 'acquired', or 'never'
        
        # Natural prompt
        natural_key = f'{origin}_natural'
        is_rep_natural, _ = generate_and_analyze(
            model, tokenizer, text, max_new_tokens, max_length, device
        )
        if is_rep_natural:
            results[natural_key]['repeating'] += 1
            results[natural_key]['indices'].append(idx)
        else:
            results[natural_key]['not_repeating'] += 1
        
        # ICL prompt
        icl_key = f'{origin}_icl'
        icl_text = format_icl_prompt(text, n_cycles)
        is_rep_icl, _ = generate_and_analyze(
            model, tokenizer, icl_text, max_new_tokens, max_length, device
        )
        if is_rep_icl:
            results[icl_key]['repeating'] += 1
            results[icl_key]['indices'].append(idx)
        else:
            results[icl_key]['not_repeating'] += 1
    
    return results

def main():
    parser = argparse.ArgumentParser(description="2x2 attention analysis: (Natural/ICL) x (Original/Acquired)")
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--evolution-file", type=str, required=True,
                       help="JSON file from cycle_evolution analysis (defines Original vs Acquired)")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-cycles", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="./outputs_2x2_analysis")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
    
    # Load origin categories (Original vs Acquired)
    print("Loading sequence categories from cycle evolution data...")
    origin_categories = load_cycle_evolution_categories(args.evolution_file, checkpoints)
    
    n_original = sum(1 for c in origin_categories.values() if c == 'original')
    n_acquired = sum(1 for c in origin_categories.values() if c == 'acquired')
    n_never = sum(1 for c in origin_categories.values() if c == 'never')
    
    print(f"Origin categories: {n_original} original, {n_acquired} acquired, {n_never} never")
    
    # Load texts
    texts = load_text_dataset(seed=args.seed, n_samples=args.n_samples)
    print(f"Loaded {len(texts)} texts")
    
    # Run 2x2 analysis
    results = run_full_2x2_analysis(
        model_name=args.model_name,
        revision=args.revision,
        origin_categories=origin_categories,
        texts=texts,
        n_cycles=args.n_cycles,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    
    # Print summary
    print("\n" + "="*70)
    print("2x2 ANALYSIS RESULTS")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.revision or 'steplatest'}")
    print("="*70)
    
    print("\n                    │   Natural   │    ICL    │")
    print("────────────────────┼─────────────┼───────────┤")
    
    for origin in ['original', 'acquired', 'never']:
        nat_key = f'{origin}_natural'
        icl_key = f'{origin}_icl'
        
        nat_total = results[nat_key]['repeating'] + results[nat_key]['not_repeating']
        icl_total = results[icl_key]['repeating'] + results[icl_key]['not_repeating']
        
        if nat_total > 0:
            nat_pct = 100 * results[nat_key]['repeating'] / nat_total
        else:
            nat_pct = 0
        
        if icl_total > 0:
            icl_pct = 100 * results[icl_key]['repeating'] / icl_total
        else:
            icl_pct = 0
        
        label = f" {origin.capitalize():17s}"
        print(f"{label}│  {results[nat_key]['repeating']:3d}/{nat_total:3d} ({nat_pct:4.1f}%) │ {results[icl_key]['repeating']:3d}/{icl_total:3d} ({icl_pct:4.1f}%) │")
    
    print("────────────────────┴─────────────┴───────────┘")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = args.revision or 'steplatest'
    safe_model = args.model_name.replace('/', '_')
    output_file = output_dir / f"analysis_2x2_{safe_model}_{checkpoint_name}.json"
    
    # Convert to serializable
    save_results = {
        'model': args.model_name,
        'checkpoint': checkpoint_name,
        'n_samples': args.n_samples,
        'origin_counts': {
            'original': n_original,
            'acquired': n_acquired,
            'never': n_never
        },
        'results': {
            k: {
                'repeating': v['repeating'],
                'not_repeating': v['not_repeating'],
                'repeating_indices': v['indices']
            }
            for k, v in results.items()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
