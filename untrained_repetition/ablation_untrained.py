import os
import torch
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from check_untrained import check_model_repetition

def empirical_ablation_sweep():
    results = []
    
    # Base configuration
    base_model = "EleutherAI/pythia-70m"
    model_sizes = [
        "EleutherAI/pythia-14m",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m"
    ]
    prompts = [
        "A", 
        "The quick brown fox jumps over the lazy dog", 
        "Random completely out of distribution text string to see if it matters."
    ]
    init_scales = [0.1, 0.5, 1.0, 2.0]
    
    # 1. Sweep Prompts (keeping pythia-70m, scale 1.0)
    print("--- 1. Sweeping Prompts ---")
    for p in prompts:
        res = check_model_repetition(base_model, prompt=p, init_scale=1.0, init_type="config_default")
        results.append({"axis": "prompt", "val": p[:20], "onset": res["onset"], "period": res["period"]})
        
    # 2. Sweep Initializations (keeping pythia-70m, prompt default)
    print("\n--- 2. Sweeping Initializations ---")
    for scale in init_scales:
        res = check_model_repetition(base_model, prompt="The quick brown fox", init_scale=scale, init_type="normal")
        results.append({"axis": "init_scale", "val": scale, "onset": res["onset"], "period": res["period"]})
        
    # 3. Sweep Model Sizes (keeping prompt default, scale 1.0)
    # This also naturally tests generation length as we set num_steps=60 which is long enough for base.
    print("\n--- 3. Sweeping Model Sizes ---")
    for m in model_sizes:
        res = check_model_repetition(m, prompt="The quick brown fox", init_scale=1.0, init_type="config_default")
        results.append({"axis": "model_size", "val": m.split('/')[-1], "onset": res["onset"], "period": res["period"]})
        
    # Save results to a clean DataFrame
    df = pd.DataFrame(results)
    df.to_csv("ablation_results.csv", index=False)
    print("\nSweep Complete! Results mapped to ablation_results.csv")
    print(df.to_string())

if __name__ == "__main__":
    empirical_ablation_sweep()
