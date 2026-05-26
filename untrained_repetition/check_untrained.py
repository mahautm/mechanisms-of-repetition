import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import argparse

def ensure_padding(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def calculate_entropy(probs):
    # compute entropy of a probability distribution
    # probs: (batch, vocab_size)
    probs = torch.clamp(probs, min=1e-10)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy

def find_repetition_period_and_onset(tokens):
    """
    Find the length of the repeating phrase and the step where it starts.
    We look for the largest repeating suffix or a clean exact period at the tail.
    """
    if not tokens:
        return 0, -1

    # Heuristic: check periods from 1 up to len(tokens)//2
    max_period_check = len(tokens) // 2
    for p in range(1, max_period_check + 1):
        suffix = tokens[-p:]
        # Check how many times this suffix repeats backwards
        repeats = 1
        while len(tokens) >= (repeats + 1) * p and tokens[-(repeats + 1)*p : -repeats*p] == suffix:
            repeats += 1
            
        if repeats > 2: # Require at least 3 repetitions of the period to be confident
            onset = len(tokens) - repeats * p
            return p, onset

    return 0, -1

def init_weights_custom(model, init_scale=1.0, init_type="normal"):
    """
    Apply custom initializations on linear layers and embeddings to see how 
    scale / type affects the onset of attractors.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if init_type == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 * init_scale)
            elif init_type == "uniform":
                torch.nn.init.uniform_(module.weight, a=-0.05 * init_scale, b=0.05 * init_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 * init_scale)

def check_model_repetition(model_name, seed=42, num_steps=60, prompt="The quick brown fox", init_scale=1.0, init_type="config_default"):
    print(f"\n--- Checking {model_name} (Init: {init_type}, Scale: {init_scale}) ---")
    torch.manual_seed(seed)
    
    # Init tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = ensure_padding(tokenizer)
    
    config = AutoConfig.from_pretrained(model_name)
    # Instantiate completely random untrained weights
    model = AutoModelForCausalLM.from_config(config)
    
    if init_type != "config_default":
        init_weights_custom(model, init_scale=init_scale, init_type=init_type)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Statistics to track
    entropies = []
    max_probs = []
    generated_tokens = []
    
    print(f"Prompt: {prompt}")
    
    with torch.no_grad():
        for step in range(num_steps):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Distribution stats
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            step_entropy = calculate_entropy(probs).item()
            step_max_prob, next_token = torch.max(probs, dim=-1)
            
            entropies.append(step_entropy)
            max_probs.append(step_max_prob.item())
            
            # Append token for next step
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            generated_tokens.append(next_token.item())
            
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    period, onset = find_repetition_period_and_onset(generated_tokens)
    print(f"Repetition metrics -> Period: {period}, Onset token index: {onset}")
    print(f"Generated text ({num_steps} steps):")
    print(">" * 20)
    print(generated_text)
    print("<" * 20)
    
    return {
        "model": model_name,
        "init_type": init_type,
        "init_scale": init_scale,
        "entropies": entropies,
        "max_probs": max_probs,
        "generated_tokens": generated_tokens,
        "period": period,
        "onset": onset,
        "label": f"{model_name.split('/')[-1]} ({init_type} {init_scale})"
    }

def plot_evolution(results):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for res in results:
        onset_idx = res["onset"]
        axes[0].plot(res["entropies"], label=res["label"], marker="o", markersize=4)
        if onset_idx >= 0:
            axes[0].axvline(onset_idx, linestyle="--", alpha=0.5, label=f"Onset {res['label']}")

        axes[1].plot(res["max_probs"], label=res["label"], marker="o", markersize=4)
        if onset_idx >= 0:
            axes[1].axvline(onset_idx, linestyle="--", alpha=0.5)
        
    axes[0].set_ylabel("Entropy")
    axes[0].set_title("Evolution of Distribution Entropy (Greedy Decoding)")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True)
    
    axes[1].set_ylabel("Max Probability")
    axes[1].set_xlabel("Generation Step")
    axes[1].set_title("Evolution of Max Probability (Greedy Decoding)")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("untrained_repetition_stats.png")
    print(f"Saved plot to untrained_repetition_stats.png")

if __name__ == "__main__":
    # We test Pythia-70m with various initializations
    # This allows us to see how initialization scales affect the repetition onset.
    tests = [
        {"model": "EleutherAI/pythia-70m", "init_type": "config_default", "scale": 1.0},
        {"model": "EleutherAI/pythia-70m", "init_type": "normal", "scale": 0.5},
        {"model": "EleutherAI/pythia-70m", "init_type": "normal", "scale": 2.0},
        {"model": "gpt2", "init_type": "config_default", "scale": 1.0},
        {"model": "TinyLlama/TinyLlama-1.1B-step-50K-105b", "init_type": "config_default", "scale": 1.0}
    ]
    
    results = []
    for t in tests:
        try:
            res = check_model_repetition(t["model"], num_steps=60, init_type=t["init_type"], init_scale=t["scale"])
            results.append(res)
        except Exception as e:
            print(f"Error checking {t['model']}: {e}")
            
    if results:
        plot_evolution(results)
