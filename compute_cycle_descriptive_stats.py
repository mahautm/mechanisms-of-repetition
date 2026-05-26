from typing import Optional
#!/usr/bin/env python3
"""Compute cycle descriptive statistics across available models for paper introduction.

Collects: cycle size, tokens-until-first-cycle, cycle count across models.
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

from parrots.cycle_detection import detect_cycles


def find_cached_hf_models() -> List[str]:
    """Try to discover locally cached Hugging Face repo ids.

    This is heuristic: look into ~/.cache/huggingface/hub/repositories for repo folders.
    """
    models = []
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "repositories"
    if cache_dir.exists():
        for repo in cache_dir.iterdir():
            if repo.is_dir():
                name = repo.name
                # replace double-dash separators with slashes
                replaced = name.replace("--", "/")
                # strip leading helper segments like 'models/' or 'repos/' if present
                if replaced.startswith("models/"):
                    replaced = replaced.split("/", 1)[1]
                if replaced.startswith("repos/"):
                    replaced = replaced.split("/", 1)[1]
                models.append(replaced)
    return models


def generate_and_detect(
    model_name: str,
    subset,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    local_files_only: bool = True,
    device_str: str = "cuda",
    device_map: Optional[str] = None,
    prompt_size: int = 512,
):
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Trying model {model_name} (local_files_only={local_files_only}) on {device}")
    try:
        # allow transformers to place model across devices if device_map is provided
        load_kwargs = {"local_files_only": local_files_only}
        if device_map is not None and device_map.lower() != "none":
            load_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = prompt_size
        # if device_map was not provided, move model to single device
        if device_map is None or device_map.lower() == "none":
            try:
                model = model.to(device)
            except Exception as e:
                print(f"  Warning: moving model to {device} failed: {e}. Falling back to CPU.")
                device = torch.device("cpu")
                model = model.to(device)
    except Exception as e:
        print(f"  Skip {model_name}: not available locally or failed to load: {e}")
        return None

    tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, 'pad_token', None) is None else tokenizer.pad_token

    outputs = []
    cycles = []
    cycle_lengths = []
    cycle_counts = []
    start_time = time.perf_counter()

    for i in range(0, len(subset), batch_size):
        batch = subset[i:i+batch_size]
        texts = list(batch["text"])
        try:
            toked = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            # move inputs to the same device as the model
            toks_device = {k: v.to(device) for k, v in toked.items()}
            toked = toks_device
            gen = model.generate(**toked, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        except Exception as e:
            print(f"  Generation failed for {model_name} on batch starting {i}: {e}")
            break

        for b in range(gen.shape[0]):
            # remove prompt tokens if present
            input_len = toked["input_ids"].shape[1]
            gen_only = gen[b, input_len:]
            c, cs, cc, _ = detect_cycles(gen_only, return_index=True)
            outputs.append(tokenizer.decode(gen[b], skip_special_tokens=True))
            cycles.append(c.cpu().numpy().tolist() if hasattr(c, 'cpu') else c)
            cycle_lengths.append(int(cs) if cs is not None else 0)
            cycle_counts.append(int(cc) if cc is not None else 0)

        del toked, gen
        torch.cuda.empty_cache()

    elapsed_seconds = time.perf_counter() - start_time

    if not outputs:
        return None

    df = pd.DataFrame({
        "model": model_name,
        "generated": outputs,
        "cycle": cycles,
        "cycle_length": cycle_lengths,
        "cycle_count": cycle_counts,
        "elapsed_seconds": elapsed_seconds,
    })
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate continuations and detect cycles (rank-aware)")
    parser.add_argument("--rank", type=int, default=0, help="Rank index (0-based)")
    parser.add_argument("--n_ranks", type=int, default=1, help="Total number of ranks")
    parser.add_argument("--max_prompts", type=int, default=1000)
    parser.add_argument("--prompt_size", type=int, default=512, help="Max prompt length for minipile inputs")
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--models",
        type=str,
        default=(
            "EleutherAI/pythia-70m,"
            "EleutherAI/pythia-1.4b,"
            "EleutherAI/pythia-6.9b,"
            "meta-llama/Llama-3.2-1B,"
            "allenai/OLMo-1B-hf,"
            "swiss-ai/Apertus-8B-2509,"
            "allenai/OLMo-2-1124-7B"
        ),
        help="Comma-separated candidate model ids",
    )
    parser.add_argument("--out_dir", type=str, default="/home/mmahaut/projects/parrots/plots")
    parser.add_argument("--allow_downloads", action="store_true", help="Allow downloading models from Hugging Face if not cached locally")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to run models on: cuda or cpu")
    parser.add_argument("--device_map", type=str, default=None, help="Device map to pass to transformers.from_pretrained (e.g., 'auto' or 'balanced'), or 'none' to disable")
    args = parser.parse_args()

    rank = args.rank
    n_ranks = args.n_ranks
    max_prompts = args.max_prompts
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    candidates = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_suffix = f"_rank{rank:03d}" if n_ranks > 1 else ""


    # add discovered cached models heuristically
    candidates += find_cached_hf_models()

    results = []
    # Load full subset once and split by rank to avoid reloading dataset per rank
    dataset = load_dataset("JeanKaddour/minipile")
    full_subset = dataset["train"].shuffle(seed=42).select(range(min(max_prompts, len(dataset["train"]))))
    full_subset = full_subset.map(lambda sample: {"text": sample["text"][: args.prompt_size]})

    subset_size = len(full_subset) // n_ranks
    start = rank * subset_size
    end = start + subset_size if rank < n_ranks - 1 else len(full_subset)
    my_subset = full_subset.select(range(start, end))

    for model_id in candidates:
        df = generate_and_detect(
            model_id,
            my_subset,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            local_files_only=not args.allow_downloads,
            device_str=args.device,
            device_map=args.device_map,
            prompt_size=args.prompt_size,
        )
        if df is None:
            continue
        # compute summary stats
        has_cycle = df["cycle_count"] > 0
        n_total = len(df)
        n_with = int(has_cycle.sum())
        pct = 100.0 * n_with / n_total if n_total > 0 else 0.0
        mean_len = float(df.loc[has_cycle, "cycle_length"].mean()) if n_with > 0 else np.nan
        std_len = float(df.loc[has_cycle, "cycle_length"].std()) if n_with > 0 else np.nan
        mean_count = float(df.loc[has_cycle, "cycle_count"].mean()) if n_with > 0 else np.nan
        std_count = float(df.loc[has_cycle, "cycle_count"].std()) if n_with > 0 else np.nan
        latency = float(df["elapsed_seconds"].iloc[0] / n_total) if n_total > 0 else np.nan

        results.append({
            "model": model_id,
            "n_total": n_total,
            "n_with_cycles": n_with,
            "pct_cyclic": pct,
            "mean_cycle_length": mean_len,
            "std_cycle_length": std_len,
            "mean_cycle_count": mean_count,
            "std_cycle_count": std_count,
            "latency": latency,
        })

        # save per-model csv
        out_dir = Path("/home/mmahaut/projects/parrots/plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            out_dir / f"cycle_detection_{re.sub('[^0-9a-zA-Z_-]', '_', model_id)}{rank_suffix}.csv",
            index=False,
        )

    if not results:
        print("No models produced results (none loaded locally).")
        return

    df_results = pd.DataFrame(results)
    out_csv = Path(f"/home/mmahaut/projects/parrots/plots/cycle_descriptive_stats{rank_suffix}.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")

    summary_lines = ["# Cycle Descriptive Statistics Summary (generated)", ""]
    for _, row in df_results.iterrows():
        summary_lines.append(f"## {row['model']}")
        summary_lines.append(f"- Total examples: {int(row['n_total'])}")
        summary_lines.append(f"- Examples with cycles: {int(row['n_with_cycles'])} ({row['pct_cyclic']:.1f}%)")
        summary_lines.append(f"- Mean cycle length: {row['mean_cycle_length']:.2f} ± {row['std_cycle_length']:.2f} tokens")
        summary_lines.append(f"- Mean cycle count: {row['mean_cycle_count']:.2f} ± {row['std_cycle_count']:.2f}")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    summary_file = Path(f"/home/mmahaut/projects/parrots/plots/CYCLE_STATS_SUMMARY{rank_suffix}.md")
    summary_file.write_text(summary_text)
    print(f"Saved markdown summary to {summary_file}")
    print(summary_text)

    latex_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\textbf{Model} &\textbf{Cyclic Seq.} & \textbf{Repetitions} & \textbf{Cycle Size} & \textbf{Cycle Number} & \textbf{Latency} \\",
        r"\hline",
    ]
    for _, row in df_results.iterrows():
        cyclic_seq = int(row["n_with_cycles"])
        repetitions = row["mean_cycle_count"] * row["n_with_cycles"] if pd.notna(row["mean_cycle_count"]) else np.nan
        cycle_size = row["mean_cycle_length"]
        cycle_number = row["mean_cycle_count"]
        latency = row["latency"]
        latex_lines.append(
            f"{row['model']} & {cyclic_seq} & "
            f"{repetitions:.2f} & {cycle_size:.2f} & {cycle_number:.2f} & {latency:.3f} \\")
    latex_lines.append(r"\end{tabular}")
    latex_table = "\n".join(latex_lines)
    latex_file = Path(f"/home/mmahaut/projects/parrots/plots/CYCLE_STATS_TABLE{rank_suffix}.tex")
    latex_file.write_text(latex_table)
    print(f"Saved LaTeX table to {latex_file}")
    print(latex_table)


if __name__ == "__main__":
    main()
