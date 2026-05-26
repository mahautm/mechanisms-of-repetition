#!/usr/bin/env python3
"""
Generate example repetitive sequences from Pythia-1.4b checkpoints
"""
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def generate_with_repetition_detection(model, tokenizer, prompt, max_new_tokens=500):
    """Generate text and detect if it becomes repetitive"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text

def main():
    checkpoints = ["step1", "step1000", "step10000", "step100000", None]  # None = final/main checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample prompts - expanded set
    prompts = [
        "The capital of France is",
        "In the year 2020,",
        "Once upon a time there was",
        "The quick brown fox",
        "Scientists have discovered that",
        "The largest mountain in the world is",
        "During the Renaissance period,",
        "The process of photosynthesis involves",
        "In quantum mechanics, the uncertainty principle states",
        "The history of computing began with",
    ]
    
    print("=" * 80)
    print("Pythia-1.4b Repetitive Sequence Examples")
    print("=" * 80)
    
    for checkpoint in checkpoints:
        print(f"\n{'='*80}")
        print(f"Checkpoint: {checkpoint if checkpoint else 'final (main)'}")
        print(f"{'='*80}\n")
        
        model_name = f"EleutherAI/pythia-1.4b"
        
        print(f"Loading model: {model_name} (revision: {checkpoint if checkpoint else 'main'})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            revision=checkpoint,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        print(f"Generating examples...\n")
        
        for i, prompt in enumerate(prompts[:5], 1):  # Show 5 examples per checkpoint
            print(f"Example {i}:")
            print(f"Prompt: '{prompt}'")
            print(f"Generated:")
            
            generated = generate_with_repetition_detection(model, tokenizer, prompt, max_new_tokens=200)
            
            # Only show the generated part (after prompt)
            prompt_length = len(prompt)
            generated_part = generated[prompt_length:]
            
            # Truncate at 500 chars for display
            if len(generated_part) > 500:
                generated_part = generated_part[:500] + "..."
            
            print(generated_part)
            print()
        
        # Clean up
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
