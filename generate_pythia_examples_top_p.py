#!/usr/bin/env python3
"""
Generate example sequences from Pythia-1.4b final checkpoint with different top_p values
"""
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def generate_with_sampling(model, tokenizer, prompt, max_new_tokens=1000, top_p=0.9):
    """Generate text with nucleus (top-p) sampling"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample prompts
    prompts = [
        "The capital of France is",
        "In the year 2020,",
        "Once upon a time there was",
    ]
    
    # Different top_p values
    top_p_values = [0.5, 0.9, 0.99]
    
    print("=" * 80)
    print("Pythia-1.4b Final Checkpoint - Top-p Sampling Comparison")
    print("=" * 80)
    
    model_name = "EleutherAI/pythia-1.4b"
    
    print(f"\nLoading model: {model_name} (final checkpoint)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*80}\n")
        
        for top_p in top_p_values:
            print(f"Top-p = {top_p}:")
            print("-" * 40)
            print(f"PROMPT: {prompt}")
            print("-" * 40)
            
            generated = generate_with_sampling(model, tokenizer, prompt, max_new_tokens=1000, top_p=top_p)
            
            # Only show the generated part (after prompt)
            prompt_length = len(prompt)
            generated_part = generated[prompt_length:]
            
            print(generated_part)
            print()
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
