from parrots.aa_fortu.aa_fortu_graphs2 import plot_from_proba
from parrots.aa_fortu.multihead_analysis_graphs import load_multihead_results_across_cycles
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path

def detect_cycle(generated_tokens, min_cycle=5, max_cycle=50, pad=0):
    cycle_length = None
    cycle = None
    for l in range(min_cycle, min(max_cycle, len(generated_tokens)//2)):
        for j in range(len(generated_tokens)-2*l):
            cycle_candidate = generated_tokens[j:j+l]
            # Ignore cycles that are all padding tokens (0)
            if all(token == pad for token in cycle_candidate):
                continue
            if cycle_candidate == generated_tokens[j+l:j+2*l]:
                cycle_length = l
                cycle = cycle_candidate
                return cycle_length, cycle
    return None, None

def process_outputs(model, inputs, texts, type, n_tokens):
    icl_inputs = {"input_ids": [], "attention_mask": []}
    results = []
    outputs = model.generate(**inputs, max_new_tokens=n_tokens)
    for i, output in enumerate(outputs):
        generated_tokens = output.tolist()
        cycle_length, cycle = detect_cycle(generated_tokens)
        if type == "natural" or cycle_length is None:
            results.append({
                "input": texts[i],
                "output": generated_tokens,
                "cycle_length": cycle_length,
                "cycle": cycle,
                "is_cyclical": cycle_length is not None
            })
        elif type == "icl":
            # we go another round of generation to get the ICL output
            icl_input = {"input_ids": [0] * (n_tokens - len(cycle)*2) + cycle * 2,
                         "attention_mask": [0] * (n_tokens - len(cycle)*2) + [1] * 2 * len(cycle)}
            icl_inputs["input_ids"].append(icl_input["input_ids"])
            icl_inputs["attention_mask"].append(icl_input["attention_mask"])
    icl_inputs = {
        "input_ids": torch.tensor(icl_inputs["input_ids"], device=inputs["input_ids"].device),
        "attention_mask": torch.tensor(icl_inputs["attention_mask"], device=inputs["attention_mask"].device)
    }
    return results, icl_inputs

def get_sorted_heads_multihead(base_path="/home/mmahaut/projects/parrots/outputs_multihead_full", 
                               model_name="EleutherAI/pythia-1.4b", 
                               checkpoints=None,
                               max_length=32,
                               natural_cycle=0,
                               icl_cycle=2,
                               no_cycle_icl_cycle=1):
    """
    Get sorted heads based on multi-head analysis across all checkpoints
    Orders heads from natural to no-cycle ICL based on contrast patterns
    
    Args:
        base_path: Base path to multihead analysis results
        model_name: Model name
        checkpoints: List of checkpoints to analyze
        max_length: Maximum sequence length
        natural_cycle: Cycle number to use for natural contrasts
        icl_cycle: Cycle number to use for ICL contrasts
        no_cycle_icl_cycle: Cycle number to use for no-cycle ICL contrasts
        
    Returns:
        dict: checkpoint -> list of (layer, head, composite_score) tuples sorted from natural to no-cycle ICL
    """
    if checkpoints is None:
        checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    
    cycle_range = [natural_cycle, icl_cycle, no_cycle_icl_cycle]
    
    print(f"Loading multi-head results for natural cycle {natural_cycle}, ICL cycle {icl_cycle}, and no-cycle ICL cycle {no_cycle_icl_cycle}")
    
    # Load the multi-head results across cycles
    results_across_cycles = load_multihead_results_across_cycles(
        base_path=base_path,
        model_name=model_name,
        checkpoints=checkpoints,
        cycle_range=cycle_range,
        max_length=max_length
    )
    
    sorted_heads_by_checkpoint = {}
    
    for checkpoint in checkpoints:
        required_categories = ['natural', 'icl', 'no_cycle_icl']
        if not all(checkpoint in results_across_cycles.get(cat, {}) for cat in required_categories):
            print(f"Warning: Checkpoint {checkpoint} not found in results")
            continue
            
        required_cycles = [natural_cycle, icl_cycle, no_cycle_icl_cycle]
        checkpoint_data = {}
        
        # Check if all required cycles are available
        missing_cycles = []
        for cat, cycle in zip(['natural', 'icl', 'no_cycle_icl'], required_cycles):
            if cycle not in results_across_cycles[cat][checkpoint]:
                missing_cycles.append(f"{cat} cycle {cycle}")
        
        if missing_cycles:
            print(f"Warning: Missing cycles for checkpoint {checkpoint}: {missing_cycles}")
            continue
        
        # Extract data for each category
        natural_data = results_across_cycles['natural'][checkpoint][natural_cycle]
        icl_data = results_across_cycles['icl'][checkpoint][icl_cycle]
        no_cycle_icl_data = results_across_cycles['no_cycle_icl'][checkpoint][no_cycle_icl_cycle]
        
        # Create comparison data for this checkpoint
        head_comparisons = []
        
        # Get all layers that have data in all three categories
        common_layers = set(natural_data.keys()) & set(icl_data.keys()) & set(no_cycle_icl_data.keys())
        
        for layer_idx in common_layers:
            natural_heads = natural_data[layer_idx]
            icl_heads = icl_data[layer_idx]
            no_cycle_icl_heads = no_cycle_icl_data[layer_idx]
            
            if any(heads is None for heads in [natural_heads, icl_heads, no_cycle_icl_heads]):
                continue
                
            # Ensure all have the same number of heads
            min_heads = min(len(natural_heads), len(icl_heads), len(no_cycle_icl_heads))
            
            for head_idx in range(min_heads):
                natural_contrast = float(natural_heads[head_idx])
                icl_contrast = float(icl_heads[head_idx])
                no_cycle_icl_contrast = float(no_cycle_icl_heads[head_idx])
                
                # Create a composite score that orders from natural to no-cycle ICL
                # Lower scores = more natural-like, Higher scores = more no-cycle ICL-like
                # Formula: (icl_contrast - natural_contrast) + 2 * (no_cycle_icl_contrast - natural_contrast)
                # This gives more weight to no-cycle ICL deviation from natural
                composite_score = (icl_contrast - natural_contrast) + 2 * (no_cycle_icl_contrast - natural_contrast)
                
                head_comparisons.append((layer_idx, head_idx, composite_score, {
                    'natural': natural_contrast,
                    'icl': icl_contrast, 
                    'no_cycle_icl': no_cycle_icl_contrast
                }))
        
        # Sort heads by composite score (ascending - natural first, no-cycle ICL last)
        sorted_heads = sorted(head_comparisons, key=lambda x: x[2], reverse=False)
        sorted_heads_by_checkpoint[checkpoint] = sorted_heads
        
        print(f"Checkpoint {checkpoint}: Found {len(sorted_heads)} heads")
        if len(sorted_heads) >= 5:
            print(f"  Top 5 most Natural-like: {[(h[0], h[1], h[2]) for h in sorted_heads[:5]]}")
            print(f"  Top 5 most No-cycle ICL-like: {[(h[0], h[1], h[2]) for h in sorted_heads[-5:]]}")
    
    return sorted_heads_by_checkpoint


def get_sorted_heads():
    """
    Legacy function - kept for backward compatibility
    Now uses the new multi-head analysis approach for a single checkpoint
    """
    sorted_heads_by_checkpoint = get_sorted_heads_multihead()
    
    # Return the latest checkpoint for backward compatibility
    if "steplatest" in sorted_heads_by_checkpoint:
        return sorted_heads_by_checkpoint["steplatest"]
    elif sorted_heads_by_checkpoint:
        # Return the first available checkpoint
        return list(sorted_heads_by_checkpoint.values())[0]
    else:
        return []


def run_ablation_across_checkpoints(base_path="/home/mmahaut/projects/parrots/outputs_multihead_full",
                                   model_name="EleutherAI/pythia-1.4b",
                                   checkpoints=None,
                                   n_samples=100,
                                   ablation_steps=None,
                                   save_results=True,
                                   benchmark_type="hellaswag"):
    """
    Run attention ablation analysis across multiple checkpoints
    
    Args:
        base_path: Base path to multihead analysis results
        model_name: Model name
        checkpoints: List of checkpoints to analyze
        n_samples: Number of samples for testing
        ablation_steps: List of numbers of heads to ablate (default: [0, 10, 20, 30, 50, 100])
        save_results: Whether to save results to JSON files
        benchmark_type: Which benchmark to use ("hellaswag", "copa", or "arc")
        
    Returns:
        dict: Results organized by checkpoint and ablation configuration
    """
    if checkpoints is None:
        checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    
    if ablation_steps is None:
        ablation_steps = [0, 10, 20, 30, 50, 100, 150, 200]
    
    # Get sorted heads for all checkpoints
    sorted_heads_by_checkpoint = get_sorted_heads_multihead(
        base_path=base_path,
        model_name=model_name,
        checkpoints=checkpoints
    )
    
    all_results = {}
    
    for checkpoint in checkpoints:
        if checkpoint not in sorted_heads_by_checkpoint:
            print(f"Skipping checkpoint {checkpoint} - no head rankings available")
            continue
            
        print(f"\n=== Running ablation for checkpoint {checkpoint} ===")
        
        sorted_heads = sorted_heads_by_checkpoint[checkpoint]
        checkpoint_results = {
            "natural_first": {"repetition": [], "icl": [], "lambada": [], "benchmark": [], "n_heads": []},
            "no_cycle_icl_first": {"repetition": [], "icl": [], "lambada": [], "benchmark": [], "n_heads": []}
        }
        
        # Test Natural-first ablation (ablate most natural-like heads first)
        print(f"Testing Natural-first ablation for {checkpoint}...")
        for n_heads in ablation_steps:
            if n_heads > len(sorted_heads):
                break
                
            print(f"  Ablating top {n_heads} Natural-like heads...")
            selected_heads = [(head[0], head[1]) for head in sorted_heads[:n_heads]]
            
            try:
                res = run_with_ablation(
                    model_name=model_name,
                    sorted_heads=selected_heads,
                    mean_heads=False,
                    n_samples=n_samples,
                    benchmark_type=benchmark_type
                )
                
                checkpoint_results["natural_first"]["repetition"].append(res["repetition"])
                checkpoint_results["natural_first"]["icl"].append(res["icl"])
                checkpoint_results["natural_first"]["lambada"].append(res["lambada"])
                checkpoint_results["natural_first"]["benchmark"].append(res["benchmark"])
                checkpoint_results["natural_first"]["n_heads"].append(n_heads)
                
            except Exception as e:
                print(f"    Error with {n_heads} heads: {e}")
                continue
        
        # Test No-cycle ICL-first ablation (ablate most no-cycle ICL-like heads first)
        print(f"Testing No-cycle ICL-first ablation for {checkpoint}...")
        reversed_heads = sorted(sorted_heads, key=lambda x: x[2], reverse=True)
        
        for n_heads in ablation_steps:
            if n_heads > len(reversed_heads):
                break
                
            print(f"  Ablating top {n_heads} No-cycle ICL-like heads...")
            selected_heads = [(head[0], head[1]) for head in reversed_heads[:n_heads]]
            
            try:
                res = run_with_ablation(
                    model_name=model_name,
                    sorted_heads=selected_heads,
                    mean_heads=False,
                    n_samples=n_samples,
                    benchmark_type=benchmark_type
                )
                
                checkpoint_results["no_cycle_icl_first"]["repetition"].append(res["repetition"])
                checkpoint_results["no_cycle_icl_first"]["icl"].append(res["icl"])
                checkpoint_results["no_cycle_icl_first"]["lambada"].append(res["lambada"])
                checkpoint_results["no_cycle_icl_first"]["benchmark"].append(res["benchmark"])
                checkpoint_results["no_cycle_icl_first"]["n_heads"].append(n_heads)
                
            except Exception as e:
                print(f"    Error with {n_heads} heads: {e}")
                continue
        
        all_results[checkpoint] = checkpoint_results
        
        # Save intermediate results
        if save_results:
            checkpoint_safe = checkpoint.replace("/", "_")
            with open(f"ablation_results_{benchmark_type}_{checkpoint_safe}.json", "w") as f:
                json.dump(checkpoint_results, f, indent=4)
    
    # Save complete results
    if save_results:
        with open(f"ablation_results_{benchmark_type}_all_checkpoints.json", "w") as f:
            json.dump(all_results, f, indent=4)
    
    return all_results


def plot_ablation_results_by_checkpoint(results, save_path=None, benchmark_type="hellaswag"):
    """
    Create plots showing ablation results across checkpoints
    
    Args:
        results: Results dictionary from run_ablation_across_checkpoints
        save_path: Path to save the plot (optional)
        benchmark_type: Type of benchmark used (for labeling)
    """
    # Prepare data for plotting
    plot_data = []
    
    for checkpoint, checkpoint_data in results.items():
        for ablation_type, type_data in checkpoint_data.items():
            n_heads_list = type_data["n_heads"]
            
            # Process each metric
            for metric in ["repetition", "icl", "lambada", "benchmark"]:
                if metric not in type_data:
                    continue
                    
                metric_data = type_data[metric]
                
                for i, n_heads in enumerate(n_heads_list):
                    if i >= len(metric_data):
                        continue
                        
                    # Extract the appropriate score
                    if metric in ["repetition", "icl"]:
                        score = metric_data[i]["cyclical_count"] / metric_data[i]["total"]
                    elif metric == "lambada":
                        score = metric_data[i]["icl_accuracy"]
                    elif metric == "benchmark":
                        score = metric_data[i]["benchmark_accuracy"]
                    
                    # Use benchmark_type for labeling instead of generic "benchmark"
                    display_metric = benchmark_type if metric == "benchmark" else metric
                    
                    plot_data.append({
                        "checkpoint": checkpoint,
                        "ablation_type": ablation_type,
                        "metric": display_metric,
                        "n_heads": n_heads,
                        "score": score
                    })
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data available for plotting")
        return
    
    # Create subplots
    metrics = df["metric"].unique()
    checkpoints = df["checkpoint"].unique()
    
    fig, axes = plt.subplots(len(metrics), len(checkpoints), 
                           figsize=(6 * len(checkpoints), 4 * len(metrics)))
    
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    if len(checkpoints) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, metric in enumerate(metrics):
        for j, checkpoint in enumerate(checkpoints):
            ax = axes[i, j] if len(metrics) > 1 else axes[j]
            
            # Filter data for this metric and checkpoint
            subset = df[(df["metric"] == metric) & (df["checkpoint"] == checkpoint)]
            
            if not subset.empty:
                sns.lineplot(data=subset, x="n_heads", y="score", 
                           hue="ablation_type", style="ablation_type", 
                           markers=True, ax=ax)
                
                ax.set_title(f"{metric.title()} - {checkpoint}")
                ax.set_xlabel("Number of Heads Ablated")
                ax.set_ylabel("Score")
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved ablation plot to {save_path}")
    
    plt.show()

def ablate_attention_heads(model, sorted_heads, mean_heads=False):
    """
    This function ablates attention heads in a model based on a list of heads to ablate.
    either zeroes out the weights or replaces them with the mean of the weights.
    this is done in the last ten lines of the function. Before that is just trying to find the attention heads in the model depending on the model architecture.

    :param model: The model to ablate.
    :param sorted_heads: A list of tuples (layer, head) indicating which heads to ablate.
    :param mean_heads: If True, replaces the weights with the mean of the weights instead of zeroing them out.
    :return: The model with the specified attention heads ablated.
    """
    # Try to find all transformer layers
    transformer_layers = []
    # Common attribute names for transformer layers
    for attr in ["transformer", "model", "gpt_neox", "gpt2"]:
        if hasattr(model, attr):
            module = getattr(model, attr)
            # Try to find layers inside the module
            for sub_attr in ["h", "layers", "decoder.layers", "block"]:
                if hasattr(module, sub_attr):
                    layers = getattr(module, sub_attr)
                    # If layers is a list or nn.ModuleList
                    if isinstance(layers, (list, torch.nn.ModuleList)):
                        transformer_layers = layers
                    elif isinstance(layers, torch.nn.Sequential):
                        transformer_layers = list(layers)
                    break
            if transformer_layers:
                break
    if not transformer_layers:
        raise ValueError("Could not find transformer layers in the model.")

    # Try to find attention module and its weights for each layer
    for layer_idx, head_idx in sorted_heads:
        layer = transformer_layers[layer_idx]
        # Try common attention attribute names
        attn = None
        for attn_attr in ["attn", "self_attn", "attention", "self_attention"]:
            if hasattr(layer, attn_attr):
                attn = getattr(layer, attn_attr)
                break
        if attn is None:
            raise ValueError(f"Could not find attention module in layer {layer_idx}.")

        # Try to find value projection weights
        w = None
        # Try common value projection attribute names
        for w_attr in ["W_v", "v_proj", "value", "value_proj", "dense_value"]:
            if hasattr(attn, w_attr):
                w = getattr(attn, w_attr)
            # If it's a Linear layer, get its weights
            if hasattr(w, "weight"):
                w = w.weight.data
            elif isinstance(w, torch.Tensor):
                w = w.data
            break
        # Special handling for GPTNeoXAttention (query_key_value or dense)
        if w is None and attn.__class__.__name__ == "GPTNeoXAttention":
            # query_key_value: [hidden_size, 3 * hidden_size]
            # We want the value part: last third of output weights
            if hasattr(attn, "query_key_value"):
                qkv = attn.query_key_value
            if hasattr(qkv, "weight"):
                qkv_weight = qkv.weight.data
            else:
                qkv_weight = qkv.data
            # Split into Q, K, V
            hidden_size = qkv_weight.shape[0]
            split_size = qkv_weight.shape[1] // 3
            # w = qkv_weight[:, 2 * split_size : 3 * split_size]
            # do it for all q k v
            w = qkv_weight
        if w is None:
            raise ValueError(f"Could not find value projection weights in attention module of layer {layer_idx}.")

        # Get number of heads
        num_heads = getattr(attn, "num_heads", None)
        if num_heads is None:
            # Try to infer from weight shape
            if hasattr(attn, "num_attention_heads"):
                num_heads = attn.num_attention_heads
            else:
                num_heads = 16
                # print(f"Warning: Could not determine number of attention heads, defaulting to {num_heads}, which is correct for Pythia 1.4b.")
                # raise ValueError("Could not determine number of attention heads.")

        head_dim = w.shape[0] // num_heads
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        with torch.no_grad():
            if mean_heads:
                mean_val = w[start:end].mean()
                w[start:end].fill_(mean_val)
            else:
                w[start:end].zero_()

    return model
def test_repetitions(model, tokenizer, type="natural", n_tokens=200, n_samples=100, prompt_size=512, seed=42, device=None):
    # type is either "natural or ICL"
    # set seed for reproducibility
    torch.manual_seed(seed)
    # Load a subset of The Pile dataset
    pile = load_dataset("JeanKaddour/minipile")
    pile = pile["train"].shuffle(seed=seed)
    pile = pile.select(range(0, n_samples))

    texts = [sample["text"][:prompt_size] for sample in pile]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=prompt_size, padding=True)
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    if device:
        model.to(device)

    results, icl_inputs = process_outputs(model, inputs, texts, type, n_tokens)

    if type == "icl":
        # with ICL we rerun with the repetitive inputs previously generated
        icl_results, _ = process_outputs(model, icl_inputs, texts, "natural", n_tokens)


    if type == "natural":
        cyclical_count = sum(1 for r in results if r["is_cyclical"])
        return {"cyclical_count": cyclical_count, "total": n_samples, "details": results}
    elif type == "icl":
        cyclical_count =  sum(1 for r in icl_results if r["is_cyclical"])
        n_icl_samples = len(icl_results)
        return {"cyclical_count": cyclical_count, "total": n_icl_samples, "details": results}

def test_lambada(model, tokenizer, n_samples=100, device=None):
    # LAMBADA is a word prediction benchmark: given a passage ("text"), the task is to predict the last word.
    # Each sample has keys: 'text' (the passage) and 'domain' (the source domain, e.g., literature).
    lambada = load_dataset("lambada", "plain_text", split="test").shuffle(seed=42).select(range(n_samples))
    contexts = []
    targets = []
    for sample in lambada:
        # 'text': the full passage, last word is the target to predict
        # 'domain': source domain (not used for prediction)
        text = sample["text"]
        # Split context and target: context is all but last word, target is last word
        parts = text.strip().split()
        if len(parts) < 2:
            continue  # skip samples too short
        context = " ".join(parts[:-1])
        target = parts[-1]
        contexts.append(context)
        targets.append(target)
    inputs = tokenizer(contexts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    if device:
        model.to(device)

    # Generate one word for each context
    outputs = model.generate(**inputs, max_new_tokens=1)
    correct = 0
    results = []
    for i, output in enumerate(outputs):
        generated = tokenizer.decode(output, skip_special_tokens=True).strip()
        # Get only the generated word(s) after the context
        gen_words = generated[len(contexts[i]):].strip().split()
        predicted = gen_words[0] if gen_words else ""
        icl_score = int(predicted == targets[i])
        correct += icl_score
        results.append({
            "context": contexts[i],
            "target": targets[i],
            "predicted": predicted,
            "output": generated,
            "icl_score": icl_score
        })
    accuracy = correct / len(results) if results else 0
    return {"icl_accuracy": accuracy, "total": len(results), "details": results}

def test_hellaswag(model, tokenizer, n_samples=100, device=None):
    """
    Test on HellaSwag dataset - requires commonsense reasoning and attention to context.
    This task is more sensitive to attention ablation than Winogrande.
    """
    hellaswag = load_dataset("hellaswag", split="validation").shuffle(seed=42).select(range(n_samples))
    correct = 0
    results = []
    
    for sample in hellaswag:
        context = sample["ctx"]
        choices = sample["endings"]
        correct_idx = int(sample["label"])
        
        # Create prompts for each choice
        choice_probs = []
        for i, choice in enumerate(choices):
            prompt = f"{context} {choice}"
            
            # Tokenize and get logits for this completion
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            model.eval()
            if device:
                model.to(device)
                
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get the probability of the completion tokens
                context_tokens = tokenizer(context, return_tensors="pt")["input_ids"]
                context_len = context_tokens.shape[1]
                
                # Calculate perplexity of the ending given the context
                ending_logits = logits[0, context_len-1:-1, :]  # logits for predicting ending tokens
                ending_tokens = inputs["input_ids"][0, context_len:]  # actual ending tokens
                
                if len(ending_tokens) > 0:
                    log_probs = torch.nn.functional.log_softmax(ending_logits, dim=-1)
                    selected_log_probs = log_probs[range(len(ending_tokens)), ending_tokens]
                    avg_log_prob = selected_log_probs.mean().item()
                    choice_probs.append(avg_log_prob)
                else:
                    choice_probs.append(float('-inf'))
        
        # Select the choice with highest probability
        predicted_idx = choice_probs.index(max(choice_probs)) if choice_probs else 0
        is_correct = predicted_idx == correct_idx
        correct += is_correct
        
        results.append({
            "context": context,
            "choices": choices,
            "correct_idx": correct_idx,
            "predicted_idx": predicted_idx,
            "choice_probs": choice_probs,
            "benchmark_score": int(is_correct)
        })
    
    accuracy = correct / n_samples
    return {"benchmark_accuracy": accuracy, "total": n_samples, "details": results}


def test_copa(model, tokenizer, n_samples=100, device=None):
    """
    Test on COPA (Choice of Plausible Alternatives) - requires causal reasoning.
    This task is very sensitive to attention mechanisms and reasoning ability.
    """
    try:
        copa = load_dataset("super_glue", "copa", split="validation").shuffle(seed=42).select(range(min(n_samples, 500)))
    except:
        # Fallback if super_glue is not available
        print("COPA dataset not available, using a smaller manual set")
        copa_samples = [
            {
                "premise": "The man broke his toe.",
                "choice1": "He got a hole in his sock.",
                "choice2": "He dropped a hammer on his foot.",
                "question": "cause",
                "label": 1
            },
            {
                "premise": "The woman felt dizzy.",
                "choice1": "She stood up quickly.",
                "choice2": "She called her doctor.",
                "question": "cause", 
                "label": 0
            }
        ] * (n_samples // 2)
        copa = copa_samples[:n_samples]
    
    correct = 0
    results = []
    
    for sample in copa:
        premise = sample["premise"]
        choice1 = sample["choice1"]
        choice2 = sample["choice2"]
        question_type = sample["question"]  # "cause" or "effect"
        correct_idx = int(sample["label"])
        
        # Create prompts based on question type
        if question_type == "cause":
            prompt1 = f"{choice1} Therefore, {premise.lower()}"
            prompt2 = f"{choice2} Therefore, {premise.lower()}"
        else:  # effect
            prompt1 = f"{premise} Therefore, {choice1.lower()}"
            prompt2 = f"{premise} Therefore, {choice2.lower()}"
        
        # Calculate likelihood for each choice
        choice_probs = []
        for prompt in [prompt1, prompt2]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            model.eval()
            if device:
                model.to(device)
                
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Calculate average log probability
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[0, :-1, :].gather(2, inputs["input_ids"][0, 1:].unsqueeze(-1)).squeeze(-1)
                avg_log_prob = token_log_probs.mean().item()
                choice_probs.append(avg_log_prob)
        
        # Select choice with higher probability
        predicted_idx = 0 if choice_probs[0] > choice_probs[1] else 1
        is_correct = predicted_idx == correct_idx
        correct += is_correct
        
        results.append({
            "premise": premise,
            "choice1": choice1,
            "choice2": choice2,
            "question_type": question_type,
            "correct_idx": correct_idx,
            "predicted_idx": predicted_idx,
            "choice_probs": choice_probs,
            "benchmark_score": int(is_correct)
        })
    
    accuracy = correct / len(results) if results else 0
    return {"benchmark_accuracy": accuracy, "total": len(results), "details": results}


def test_benchmarks(model, tokenizer, n_samples=100, device=None, benchmark_type="hellaswag"):
    """
    Test on various benchmarks that are sensitive to attention ablation.
    
    Available benchmarks:
    - hellaswag: Commonsense reasoning, context-dependent
    - copa: Causal reasoning, very attention-sensitive
    - arc: Knowledge-based reasoning (original, kept for compatibility)
    """
    if benchmark_type == "hellaswag":
        return test_hellaswag(model, tokenizer, n_samples, device)
    elif benchmark_type == "copa":
        return test_copa(model, tokenizer, n_samples, device)
    elif benchmark_type == "arc":
        # Keep original ARC implementation for compatibility
        arc = load_dataset("ai2_arc", "ARC-Challenge", split="test").shuffle(seed=42).select(range(n_samples))
        prompts = []
        answers = []
        for sample in arc:
            question = sample["question"]
            choices = sample["choices"]["text"]
            answer = sample["answerKey"]
            prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
            prompts.append(prompt)
            answers.append(answer)
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        model.eval()
        if device:
            model.to(device)

        outputs = model.generate(**inputs, max_new_tokens=5)
        correct = 0
        results = []
        for i, output in enumerate(outputs):
            generated = tokenizer.decode(output, skip_special_tokens=True)
            pred = generated.strip().split()[0].upper() if generated.strip() else ""
            benchmark_score = int(pred == answers[i])
            correct += benchmark_score
            results.append({
                "prompt": prompts[i],
                "expected": answers[i],
                "output": generated,
                "benchmark_score": benchmark_score
            })
        accuracy = correct / n_samples
        return {"benchmark_accuracy": accuracy, "total": n_samples, "details": results}
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

def run_with_ablation(
    model_name: str = "EleutherAI/pythia-1.4b",
    sorted_heads: list = None,
    mean_heads: bool = False,
    n_samples: int = 100,
    benchmark_type: str = "hellaswag"
    ):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # add PAD token for better handling of special tokens
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ablate attention heads
    model = ablate_attention_heads(model, sorted_heads, mean_heads=mean_heads)

    repetition_results = test_repetitions(model, tokenizer, type="natural", n_tokens=100, n_samples=n_samples, prompt_size=100, seed=42, device=device)
    print(f"Repetition Cyclical Count: {repetition_results['cyclical_count']} / {repetition_results['total']}")
    icl_results = test_repetitions(model, tokenizer, type="icl", n_tokens=100, n_samples=n_samples, prompt_size=100, seed=42, device=device)
    print(f"ICL Cyclical Count: {icl_results['cyclical_count']} / {icl_results['total']}")
    lambada_results = test_lambada(model, tokenizer, n_samples=n_samples, device=device)
    print(f"LAMBADA ICL Accuracy: {lambada_results['icl_accuracy']:.3f}")
    benchmark_results = test_benchmarks(model, tokenizer, n_samples=n_samples, device=device, benchmark_type=benchmark_type)
    print(f"{benchmark_type.upper()} Accuracy: {benchmark_results['benchmark_accuracy']:.3f}")

    # Return all results for further analysis
    return {
        "repetition": repetition_results,
        "icl": icl_results,
        "lambada": lambada_results,
        "benchmark": benchmark_results,
        "benchmark_type": benchmark_type
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run attention ablation analysis")
    parser.add_argument("--run_experiments", action="store_true", 
                       help="Run ablation experiments")
    parser.add_argument("--plot_results", action="store_true", 
                       help="Plot existing results")
    parser.add_argument("--run_across_checkpoints", action="store_true",
                       help="Run ablation across all checkpoints")
    parser.add_argument("--n_samples", type=int, default=100,
                       help="Number of samples for testing")
    parser.add_argument("--checkpoints", nargs="+", 
                       default=["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"],
                       help="Checkpoints to analyze")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1.4b",
                       help="Model name")
    parser.add_argument("--base_path", type=str, default="/home/mmahaut/projects/parrots/outputs_multihead_full",
                       help="Base path to multihead analysis results")
    parser.add_argument("--benchmark", type=str, default="hellaswag", 
                       choices=["hellaswag", "copa", "arc"],
                       help="Benchmark to use for evaluation (hellaswag=commonsense, copa=causal reasoning, arc=knowledge)")
    
    args = parser.parse_args()
    
    print(f"Using benchmark: {args.benchmark}")
    if args.benchmark == "hellaswag":
        print("  - HellaSwag: Commonsense reasoning, context-dependent (most attention-sensitive)")
    elif args.benchmark == "copa":
        print("  - COPA: Causal reasoning, very attention-sensitive")
    elif args.benchmark == "arc":
        print("  - ARC: Knowledge-based reasoning (original benchmark)")
    
    if args.run_across_checkpoints:
        print("Running ablation analysis across all checkpoints...")
        results = run_ablation_across_checkpoints(
            base_path=args.base_path,
            model_name=args.model_name,
            checkpoints=args.checkpoints,
            n_samples=args.n_samples,
            benchmark_type=args.benchmark
        )
        
        print("Creating comprehensive plots...")
        plot_ablation_results_by_checkpoint(
            results, 
            save_path=f"ablation_results_{args.benchmark}_all_checkpoints.png",
            benchmark_type=args.benchmark
        )
        
    elif args.run_experiments:
        print("Running single checkpoint ablation experiments...")
        sorted_heads = get_sorted_heads()
        print(f"Top 10 heads: {sorted_heads[:10]}")
        
        # Get number of heads that have different composite scores
        heads_natural_like = [head for head in sorted_heads if head[2] < 0]  # Negative composite score = more natural
        heads_no_cycle_icl_like = [head for head in sorted_heads if head[2] > 0]  # Positive composite score = more no-cycle ICL
        print(f"Number of heads more Natural-like (score < 0): {len(heads_natural_like)}")
        print(f"Number of heads more No-cycle ICL-like (score > 0): {len(heads_no_cycle_icl_like)}")
        print(f"Score range: {sorted_heads[0][2]:.3f} (most natural) to {sorted_heads[-1][2]:.3f} (most no-cycle ICL)")
        
        # Show contrast details for top heads
        print("\nTop 5 most Natural-like heads (lowest composite scores):")
        for i, (layer, head, score, contrasts) in enumerate(sorted_heads[:5]):
            print(f"  {i+1}. Layer {layer}, Head {head}: score={score:.3f}")
            print(f"     Natural={contrasts['natural']:.3f}, ICL={contrasts['icl']:.3f}, No-cycle ICL={contrasts['no_cycle_icl']:.3f}")
        
        print("\nTop 5 most No-cycle ICL-like heads (highest composite scores):")
        for i, (layer, head, score, contrasts) in enumerate(sorted_heads[-5:]):
            print(f"  {i+1}. Layer {layer}, Head {head}: score={score:.3f}")
            print(f"     Natural={contrasts['natural']:.3f}, ICL={contrasts['icl']:.3f}, No-cycle ICL={contrasts['no_cycle_icl']:.3f}")
        
        results = {}
        
        # Experiment 1: Gradually ablate Natural-like heads (start from most natural)
        results["natural_first"] = {"repetition": [], "icl": [], "lambada": [], "benchmark": [], "n_heads": []}
        
        for n_heads in [0, 10, 20, 30, 50, 100, 150, 200]:
            print(f"Ablating top {n_heads} Natural-like heads")
            selected_heads = [head[:2] for head in sorted_heads[:n_heads]]
            res = run_with_ablation(sorted_heads=selected_heads, mean_heads=False, 
                                  n_samples=args.n_samples, benchmark_type=args.benchmark)
            results["natural_first"]["repetition"].append(res["repetition"])
            results["natural_first"]["icl"].append(res["icl"])
            results["natural_first"]["lambada"].append(res["lambada"])
            results["natural_first"]["benchmark"].append(res["benchmark"])
            results["natural_first"]["n_heads"].append(n_heads)
        
        # Experiment 2: Gradually ablate No-cycle ICL-like heads (start from most no-cycle ICL)
        results["no_cycle_icl_first"] = {"repetition": [], "icl": [], "lambada": [], "benchmark": [], "n_heads": []}
        sorted_heads_reversed = sorted(sorted_heads, key=lambda x: x[2], reverse=True)
        
        for n_heads in [0, 10, 20, 30, 50, 100, 150, 200]:
            print(f"Ablating top {n_heads} No-cycle ICL-like heads")
            selected_heads = [head[:2] for head in sorted_heads_reversed[:n_heads]]
            res = run_with_ablation(sorted_heads=selected_heads, mean_heads=False, 
                                  n_samples=args.n_samples, benchmark_type=args.benchmark)
            results["no_cycle_icl_first"]["repetition"].append(res["repetition"])
            results["no_cycle_icl_first"]["icl"].append(res["icl"])
            results["no_cycle_icl_first"]["lambada"].append(res["lambada"])
            results["no_cycle_icl_first"]["benchmark"].append(res["benchmark"])
            results["no_cycle_icl_first"]["n_heads"].append(n_heads)
        
        # Save results
        with open(f"ablation_results_{args.benchmark}_natural_first.json", "w") as f:
            json.dump(results["natural_first"], f, indent=4)
        with open(f"ablation_results_{args.benchmark}_no_cycle_icl_first.json", "w") as f:
            json.dump(results["no_cycle_icl_first"], f, indent=4)
        with open(f"ablation_results_{args.benchmark}.json", "w") as f:
            json.dump(results, f, indent=4)
    
    elif args.plot_results:
        print("Plotting existing results...")
        benchmark_found = False
        
        # Try to load comprehensive results for specified benchmark
        try:
            with open(f"ablation_results_{args.benchmark}_all_checkpoints.json", "r") as f:
                all_results = json.load(f)
            plot_ablation_results_by_checkpoint(
                all_results, 
                save_path=f"ablation_results_{args.benchmark}_all_checkpoints.png",
                benchmark_type=args.benchmark
            )
            benchmark_found = True
            
        except FileNotFoundError:
            print(f"No comprehensive results found for {args.benchmark}, trying single checkpoint results...")
            
            try:
                with open(f"ablation_results_{args.benchmark}_natural_first.json", "r") as f:
                    results_natural_first = json.load(f)
                with open(f"ablation_results_{args.benchmark}_no_cycle_icl_first.json", "r") as f:
                    results_no_cycle_icl_first = json.load(f)
                
                results = {"natural_first": results_natural_first, "no_cycle_icl_first": results_no_cycle_icl_first}
                
                plot_data = []
                
                for ablation_type, type_data in results.items():
                    n_heads_list = type_data["n_heads"]
                    
                    # Process each metric
                    for metric in ["repetition", "icl", "lambada", "benchmark"]:
                        if metric not in type_data:
                            continue
                            
                        metric_data = type_data[metric]
                        for i, n_heads in enumerate(n_heads_list):
                            if metric in ["repetition", "icl"]:
                                score = metric_data[i]["cyclical_count"] / metric_data[i]["total"]
                            elif metric == "lambada":
                                score = metric_data[i]["icl_accuracy"]
                            elif metric == "benchmark":
                                score = metric_data[i]["benchmark_accuracy"]
                            
                            display_metric = args.benchmark if metric == "benchmark" else metric
                            
                            plot_data.append({
                                "ablation_type": ablation_type,
                                "metric": display_metric,
                                "n_heads": n_heads,
                                "score": score
                            })
                
                # Convert to DataFrame and plot
                df = pd.DataFrame(plot_data)
                
                if not df.empty:
                    metrics = df["metric"].unique()
                    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
                    
                    if len(metrics) == 1:
                        axes = [axes]
                    
                    for i, metric in enumerate(metrics):
                        subset = df[df["metric"] == metric]
                        sns.lineplot(data=subset, x="n_heads", y="score", hue="ablation_type", 
                                   style="ablation_type", markers=True, ax=axes[i])
                        axes[i].set_title(f"{metric.title()}")
                        axes[i].set_xlabel("Number of Heads Ablated")
                        axes[i].set_ylabel("Score")
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f"ablation_results_{args.benchmark}.png", bbox_inches='tight', dpi=300)
                    plt.show()
                    benchmark_found = True
                    
            except FileNotFoundError:
                pass
        
        if not benchmark_found:
            print(f"No ablation results found for benchmark '{args.benchmark}'. Run experiments first with --run_experiments or --run_across_checkpoints")
            print(f"Available benchmark options: hellaswag, copa, arc")
    
    else:
        print("No action specified. Use --run_experiments, --run_across_checkpoints, or --plot_results")
        print("Available benchmark options:")
        print("  --benchmark hellaswag  (recommended: commonsense reasoning, most attention-sensitive)")
        print("  --benchmark copa       (causal reasoning, very attention-sensitive)")
        print("  --benchmark arc        (knowledge-based reasoning, original)")
        print("")
        print("Usage examples:")
        print("  python attention_ablation.py --run_across_checkpoints --benchmark hellaswag")
        print("  python attention_ablation.py --run_experiments --benchmark copa --n_samples 50")
        print("  python attention_ablation.py --plot_results --benchmark hellaswag")