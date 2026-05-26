import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def ensure_padding(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def find_repetition_period_and_onset(tokens):
    if not tokens:
        return 0, -1

    max_period_check = len(tokens) // 2
    for p in range(1, max_period_check + 1):
        suffix = tokens[-p:]
        repeats = 1
        while len(tokens) >= (repeats + 1) * p and tokens[-(repeats + 1)*p : -repeats*p] == suffix:
            repeats += 1
            
        if repeats > 2: # Require at least 3 repetitions of the period
            onset = len(tokens) - repeats * p
            return p, onset

    return 0, -1

def generate_and_measure(model, tokenizer, prompt_text, device, max_new_tokens=40):
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    
    # To avoid OOM and excessive compute, we keep track of things step by step
    metrics = {
        "residual_ratios": [],
        "attention_entropies": [],
        "cosine_sims": [],
        "tokens": []
    }
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True, output_attentions=True)
            hs = outputs.hidden_states
            last_idx = -1
            
            # 1. Residual Norm Ratio
            layer_ratios = []
            for l in range(len(hs) - 1):
                x_in = hs[l][0, last_idx, :]
                x_out = hs[l+1][0, last_idx, :]
                residual = x_out - x_in
                ratio = torch.norm(residual) / (torch.norm(x_in) + 1e-8)
                layer_ratios.append(ratio.item())
            metrics["residual_ratios"].append(np.mean(layer_ratios))
            
            # 2. Attention Entropy
            layer_entropies = []
            for attn in outputs.attentions:
                probs = attn[0, :, last_idx, :]
                probs = torch.clamp(probs, min=1e-10)
                entropy = -torch.sum(probs * torch.log(probs), dim=-1)
                layer_entropies.append(entropy.mean().item())
            metrics["attention_entropies"].append(np.mean(layer_entropies))
            
            # 3. Contextual Rank Collapse (Cosine Sim with previous token state)
            curr_rep = hs[-1][0, last_idx, :]
            prev_rep = hs[-1][0, last_idx - 1, :]
            cos_sim = torch.nn.functional.cosine_similarity(curr_rep.unsqueeze(0), prev_rep.unsqueeze(0)).item()
            metrics["cosine_sims"].append(cos_sim)
            
            # Next token (greedy)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            metrics["tokens"].append(next_token.item())
            
    # Find repetition
    period, onset = find_repetition_period_and_onset(metrics["tokens"])
    metrics["onset"] = onset
    metrics["period"] = period
    
    return metrics

def plot_aligned_metrics(repeating, control, window=15):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    metrics_keys = [("residual_ratios", "Residual Update Norm Ratio"), 
                    ("attention_entropies", "Attention Entropy (Bits)"), 
                    ("cosine_sims", "Contextual Rank Collapse (Cosine Sim $x_t, x_{t-1}$)")]

    x_axis = np.arange(-window, window + 1)
    
    for ax, (m_key, title) in zip(axes, metrics_keys):
        # Align repeating
        rep_aligned = []
        for r in repeating:
            onset = r["onset"]
            if onset is not None and onset >= window and onset + window < len(r[m_key]):
                rep_aligned.append(r[m_key][onset - window : onset + window + 1])
                
        if rep_aligned:
            rep_arr = np.array(rep_aligned)
            rep_mean = rep_arr.mean(axis=0)
            rep_std = rep_arr.std(axis=0)
            ax.plot(x_axis, rep_mean, label="Repeating Trajectories", color='red')
            ax.fill_between(x_axis, rep_mean - rep_std, rep_mean + rep_std, color='red', alpha=0.2)
            
        # Align control (just take chunks of length 2*window + 1 from the middle of generation)
        ctrl_aligned = []
        for c in control:
            if len(c[m_key]) >= 2 * window + 1:
                mid = len(c[m_key]) // 2
                ctrl_aligned.append(c[m_key][mid - window : mid + window + 1])
                
        if ctrl_aligned:
            ctrl_arr = np.array(ctrl_aligned)
            ctrl_mean = ctrl_arr.mean(axis=0)
            ctrl_std = ctrl_arr.std(axis=0)
            ax.plot(x_axis, ctrl_mean, label="Control (Non-Repeating)", color='blue')
            ax.fill_between(x_axis, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color='blue', alpha=0.2)
            
        ax.axvline(0, color='black', linestyle='--', label="Repetition Onset (t=0)")
        ax.set_ylabel(m_key)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
    axes[2].set_xlabel("Tokens relative to Repetition Onset")
    plt.tight_layout()
    plt.savefig("uncertain_attractor_proof.png")
    print("Saved plot to uncertain_attractor_proof.png")

def main():
    model_name = "EleutherAI/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = ensure_padding(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    print("Loading Minipile dataset...")
    dataset = load_dataset("JeanKaddour/minipile", split="train")
    
    repeating_trajectories = []
    control_trajectories = []
    
    # We need highly repetitive ones. So we just generate on short prompts.
    num_samples_to_try = 50
    valid_window = 15 # we need at least 'window' tokens before and after onset to plot
    
    # We format slightly weird chunks of text to encourage repeating out-of-distribution states
    print("Harvesting trajectories...")
    for i in tqdm(range(num_samples_to_try)):
        text = dataset[i]["text"]
        # Use a short prefix (16 tokens)
        tokens = tokenizer.encode(text, truncation=True, max_length=16)
        prompt = tokenizer.decode(tokens)
        
        metrics = generate_and_measure(model, tokenizer, prompt, device, max_new_tokens=40)
        
        onset = metrics["onset"]
        if onset > valid_window and onset + valid_window < len(metrics["tokens"]):
            repeating_trajectories.append(metrics)
        else:
            # If it didn't repeat neatly within bounds, use as control
            if onset == -1: 
                control_trajectories.append(metrics)
                
    print(f"Harvested {len(repeating_trajectories)} repeating and {len(control_trajectories)} control trajectories.")
    
    plot_aligned_metrics(repeating_trajectories, control_trajectories, window=valid_window)

if __name__ == "__main__":
    main()
