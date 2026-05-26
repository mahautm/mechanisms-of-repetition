#!/usr/bin/env python3
"""
Generate repetition tracking data in the same format as ckpt_pipeline_main.py
but WITHOUT requiring trained lenses (only for alluvial plots)

This script tracks:
- data index: all datapoints processed  
- repetition index: which datapoints are repeating (natural)
- no-cycle icl index: which datapoints are repeating with cycle-free input
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from parrots.cycle_detection import detect_cycles
from parrots.aa_fortu.modules.data_utils import load_text_dataset

def generate_and_detect(model, tokenizer, text, max_new_tokens=1000, max_length=0, device='cuda'):
    """Generate text and detect if it cycles

    Important: when the model is sharded across multiple GPUs via `device_map`,
    moving the inputs to a single `device` (e.g. 'cuda') can create a hotspot
    on GPU0 and cause CUDA OOM. Keep inputs on CPU in that case and let the
    transformers runtime handle routing to shards.
    """
    # Decide where to place input tensors. If the model has a device map (i.e.
    # was loaded with `device_map='auto'`), keep inputs on CPU to avoid filling
    # a single GPU with activations. Otherwise move inputs to the chosen device.
    has_device_map = False
    # transformers stores hf_device_map or device_map depending on version
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        has_device_map = True
    elif hasattr(model, 'device_map') and model.device_map:
        has_device_map = True

    if max_length and max_length > 0:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=False)

    if not has_device_map:
        inputs = inputs.to(device)
    
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
    return is_repeating, cycle_size, cycle_count

def format_icl_prompt(text, n_cycles=4):
    """Create ICL-style prompt with repeated examples"""
    # Simple ICL format: repeat the text n_cycles times
    return (text + "\n") * n_cycles + text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--sample-offset", type=int, default=0, help="Start index offset for samples (for batching)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--n-cycles", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./outputs_multihead_full")
    parser.add_argument("--layer", type=int, default=4, help="Layer index (for output path compatibility)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", type=str, default=None, help="Transformers device_map setting such as 'auto' to shard across multiple GPUs")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model {args.model_name}...")
    load_kwargs = {"torch_dtype": torch.float16}
    if args.device_map and args.device_map.lower() != 'none':
        load_kwargs["device_map"] = args.device_map
    if args.revision and args.revision != 'steplatest':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            revision=args.revision,
            **load_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **load_kwargs
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if not args.device_map or args.device_map.lower() == 'none':
        model.to(device)
    model.eval()
    
    # Load dataset (allow requesting offset+n_samples then slice)
    total_required = args.sample_offset + args.n_samples
    print(f"Loading {total_required} text samples (will slice offset {args.sample_offset}..)")
    texts = load_text_dataset(seed=args.seed, n_samples=total_required)
    # Slice to requested window (supports batching across tasks)
    texts = texts[args.sample_offset: args.sample_offset + args.n_samples]
    
    # Track results
    data_indices = []
    repetition_indices = []
    no_cycle_icl_indices = []

    print("Processing texts...")
    for idx, text in enumerate(tqdm(texts)):
        data_indices.append(idx)
        
        # Natural prompt (no ICL)
        is_repeating, _, _ = generate_and_detect(
            model, tokenizer, text,
            args.max_new_tokens, args.max_length, device
        )
        
        if is_repeating:
            repetition_indices.append(idx)
        
        # ICL prompt (with cycles in input)
        icl_text = format_icl_prompt(text, args.n_cycles)
        is_icl_repeating, _, _ = generate_and_detect(
            model, tokenizer, icl_text,
            args.max_new_tokens, args.max_length, device
        )
        
        # no-cycle ICL = not repeating with natural, but repeating with ICL
        # OR: repeating with ICL style input but without the cycle
        # Actually looking at the original code, no-cycle-icl means:
        # sequences that do NOT have cycles in the output when using ICL-style prompts
        if not is_icl_repeating:
            no_cycle_icl_indices.append(idx)
    
    # Create output directory
    checkpoint_name = args.revision if args.revision else "steplatest"
    output_dir = Path(args.output_dir) / args.model_name / checkpoint_name / f"layer_{args.layer}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    length_tag = f"ml{args.max_length}" if args.max_length and args.max_length > 0 else "full"
    # Include sample offset in filename when batching so outputs don't overwrite
    offset_tag = f"off{args.sample_offset}" if args.sample_offset and args.sample_offset > 0 else "off0"
    output_file = output_dir / f"full_analysis_cyc{args.n_cycles}_{length_tag}_{offset_tag}.out"
    
    # Write output in same format as ckpt_pipeline_main.py
    with open(output_file, 'w') as f:
        f.write(f"layer {args.layer} cycle count: {len(repetition_indices)}\n")
        f.write(f"layer {args.layer} data index: {data_indices}\n")
        f.write(f"layer {args.layer} repetition index: {repetition_indices}\n")
        f.write(f"layer {args.layer} no-cycle icl index: {no_cycle_icl_indices}\n")
        f.write(f"layer {args.layer} no-cycle icl cycle count: {len(no_cycle_icl_indices)}\n")
    
    print(f"\nResults:")
    print(f"  Total samples: {len(data_indices)}")
    print(f"  Natural repeating: {len(repetition_indices)} ({100*len(repetition_indices)/len(data_indices):.1f}%)")
    print(f"  No-cycle ICL: {len(no_cycle_icl_indices)} ({100*len(no_cycle_icl_indices)/len(data_indices):.1f}%)")
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    main()
