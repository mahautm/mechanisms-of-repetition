#!/usr/bin/env python3
"""
Head-by-head, cycle-by-cycle ablation analysis across training checkpoints.

This script provides a causal alternative to probe-based head analysis by
ablating one attention head at a time and measuring changes in repetition
cycles during generation.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from parrots.cycle_detection import detect_cycles


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def get_transformer_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError("Unsupported model architecture: cannot find transformer layers")


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attn"):
        return layer.attn
    raise ValueError("Unsupported layer type: cannot find attention module")


def infer_num_heads(model, attention_module: torch.nn.Module) -> int:
    for attr in ["num_heads", "n_heads", "num_attention_heads", "n_head"]:
        if hasattr(attention_module, attr):
            value = getattr(attention_module, attr)
            if isinstance(value, int) and value > 0:
                return value
    if hasattr(model.config, "num_attention_heads"):
        return int(model.config.num_attention_heads)
    raise ValueError("Could not infer number of attention heads")


def load_text_prompts(n_samples: int, seed: int, dataset_name: str) -> List[str]:
    if dataset_name.lower() == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if isinstance(t, str) and len(t.split()) > 8]
    else:
        try:
            from datasets import load_dataset

            ds = load_dataset(dataset_name, split="train")
            key = "text" if "text" in ds.column_names else ds.column_names[0]
            texts = [t for t in ds[key] if isinstance(t, str) and len(t.split()) > 8]
        except Exception:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"] if isinstance(t, str) and len(t.split()) > 8]

    random.Random(seed).shuffle(texts)
    return texts[:n_samples]


def run_generation(
    model,
    tokenizer,
    prompts: List[str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[Dict]:
    device = next(model.parameters()).device
    rows = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)

        prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if do_sample:
            generate_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
        else:
            generate_kwargs.update(dict(do_sample=False))

        with torch.no_grad():
            out = model.generate(**enc, **generate_kwargs)

        for idx_in_batch, prompt in enumerate(batch_prompts):
            output_ids = out[idx_in_batch]
            prompt_len = int(prompt_lengths[idx_in_batch])
            gen_ids = output_ids[prompt_len:].tolist()
            cycle, cycle_size, cycle_count = detect_cycles(gen_ids)

            rows.append(
                {
                    "prompt": prompt,
                    "prompt_index": start + idx_in_batch,
                    "has_cycle": int(cycle is not None and cycle_count > 1 and cycle_size > 0),
                    "cycle_size": int(cycle_size) if cycle_size is not None else 0,
                    "cycle_count": int(cycle_count) if cycle_count is not None else 0,
                }
            )

    return rows


def create_head_ablation_hook(head_idx: int, num_heads: int):
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            if not output:
                return output
            attn_out = output[0]
            remainder = output[1:]
        else:
            attn_out = output
            remainder = None

        if not torch.is_tensor(attn_out) or attn_out.dim() != 3:
            return output

        hidden_size = attn_out.shape[-1]
        if hidden_size % num_heads != 0:
            return output

        head_dim = hidden_size // num_heads
        if head_idx >= num_heads:
            return output

        start = head_idx * head_dim
        end = start + head_dim

        modified = attn_out.clone()
        modified[..., start:end] = 0.0

        if remainder is None:
            return modified
        return (modified, *remainder)

    return hook_fn


def summarize(df: pd.DataFrame, cycle_sizes: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        df.groupby(["checkpoint", "layer", "head", "condition"], as_index=False)
        .agg(
            n_prompts=("prompt_index", "count"),
            repetition_rate=("has_cycle", "mean"),
            avg_cycle_count=("cycle_count", "mean"),
        )
        .sort_values(["checkpoint", "layer", "head", "condition"])
    )

    cycle_rows = []
    keys = ["checkpoint", "layer", "head", "condition"]
    for key, part in df.groupby(keys):
        checkpoint, layer, head, condition = key
        total = len(part)
        for k in cycle_sizes:
            cycle_rows.append(
                {
                    "checkpoint": checkpoint,
                    "layer": layer,
                    "head": head,
                    "condition": condition,
                    "cycle_size": k,
                    "rate_cycle_size_eq_k": float((part["cycle_size"] == k).mean()),
                    "count_cycle_size_eq_k": int((part["cycle_size"] == k).sum()),
                    "n_prompts": total,
                }
            )

    cycle_df = pd.DataFrame(cycle_rows)

    baseline_overall = overall[overall["condition"] == "baseline"].rename(
        columns={
            "repetition_rate": "baseline_repetition_rate",
            "avg_cycle_count": "baseline_avg_cycle_count",
        }
    )
    ablated_overall = overall[overall["condition"] == "ablated"].rename(
        columns={
            "repetition_rate": "ablated_repetition_rate",
            "avg_cycle_count": "ablated_avg_cycle_count",
        }
    )

    merged_overall = baseline_overall.merge(
        ablated_overall,
        on=["checkpoint", "layer", "head", "n_prompts"],
        how="inner",
    )
    merged_overall["delta_repetition_rate"] = (
        merged_overall["ablated_repetition_rate"] - merged_overall["baseline_repetition_rate"]
    )
    merged_overall["delta_avg_cycle_count"] = (
        merged_overall["ablated_avg_cycle_count"] - merged_overall["baseline_avg_cycle_count"]
    )

    base_cycle = cycle_df[cycle_df["condition"] == "baseline"].rename(
        columns={"rate_cycle_size_eq_k": "baseline_rate_cycle_size_eq_k"}
    )
    abl_cycle = cycle_df[cycle_df["condition"] == "ablated"].rename(
        columns={"rate_cycle_size_eq_k": "ablated_rate_cycle_size_eq_k"}
    )

    merged_cycle = base_cycle.merge(
        abl_cycle,
        on=["checkpoint", "layer", "head", "cycle_size", "n_prompts"],
        how="inner",
        suffixes=("_baseline", "_ablated"),
    )
    merged_cycle["delta_rate_cycle_size_eq_k"] = (
        merged_cycle["ablated_rate_cycle_size_eq_k"] - merged_cycle["baseline_rate_cycle_size_eq_k"]
    )

    return merged_overall, merged_cycle


def main():
    parser = argparse.ArgumentParser(
        description="Head-by-head, cycle-by-cycle ablation analysis across checkpoints"
    )
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=["step1", "step1000", "step5000", "step10000", "step100000", "steplatest"],
    )
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument(
        "--heads",
        type=str,
        default="",
        help="Comma-separated head indices. Empty means all heads.",
    )
    parser.add_argument("--n_samples", type=int, default=120)
    parser.add_argument("--dataset", type=str, default="JeanKaddour/minipile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cycle_sizes", type=str, default="2,3,4,5,6")
    parser.add_argument("--output_dir", type=str, default="./outputs_ablation_head_cycle")

    args = parser.parse_args()
    set_seed(args.seed)

    cycle_sizes = parse_int_list(args.cycle_sizes)
    prompts = load_text_prompts(args.n_samples, args.seed, args.dataset)
    if not prompts:
        raise ValueError("No prompts loaded; cannot run ablation analysis")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for checkpoint in args.checkpoints:
        print("=" * 90)
        print(f"Checkpoint: {checkpoint}")
        print("=" * 90)

        revision: Optional[str] = None if checkpoint == "steplatest" else checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            revision=revision,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            revision=revision,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        layers = get_transformer_layers(model)
        if args.layer < 0 or args.layer >= len(layers):
            raise ValueError(f"Invalid layer index {args.layer}; model has {len(layers)} layers")

        attention_module = get_attention_module(layers[args.layer])
        num_heads = infer_num_heads(model, attention_module)

        requested_heads = parse_int_list(args.heads)
        if requested_heads:
            heads = [h for h in requested_heads if 0 <= h < num_heads]
        else:
            heads = list(range(num_heads))

        print(f"Layer {args.layer} with {num_heads} heads; evaluating {len(heads)} heads")

        baseline_rows = run_generation(
            model,
            tokenizer,
            prompts,
            args.max_prompt_tokens,
            args.max_new_tokens,
            args.batch_size,
            args.do_sample,
            args.temperature,
            args.top_p,
        )
        for row in baseline_rows:
            row.update(
                {
                    "checkpoint": checkpoint,
                    "layer": args.layer,
                    "head": -1,
                    "condition": "baseline",
                }
            )
        all_rows.extend(baseline_rows)

        for head_idx in tqdm(heads, desc=f"Heads ({checkpoint})"):
            hook = create_head_ablation_hook(head_idx=head_idx, num_heads=num_heads)
            handle = attention_module.register_forward_hook(hook)
            try:
                ablated_rows = run_generation(
                    model,
                    tokenizer,
                    prompts,
                    args.max_prompt_tokens,
                    args.max_new_tokens,
                    args.batch_size,
                    args.do_sample,
                    args.temperature,
                    args.top_p,
                )
            finally:
                handle.remove()

            for row in ablated_rows:
                row.update(
                    {
                        "checkpoint": checkpoint,
                        "layer": args.layer,
                        "head": head_idx,
                        "condition": "ablated",
                    }
                )
            all_rows.extend(ablated_rows)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_df = pd.DataFrame(all_rows)

    baseline_df = raw_df[raw_df["condition"] == "baseline"].copy()
    ablated_df = raw_df[raw_df["condition"] == "ablated"].copy()

    expanded_baseline = []
    for _, row in baseline_df.iterrows():
        for head in sorted(ablated_df["head"].unique()):
            replicated = row.to_dict()
            replicated["head"] = int(head)
            expanded_baseline.append(replicated)
    baseline_expanded_df = pd.DataFrame(expanded_baseline)

    aligned_df = pd.concat([baseline_expanded_df, ablated_df], ignore_index=True)
    overall_delta_df, cycle_delta_df = summarize(aligned_df, cycle_sizes)

    safe_model = args.model_name.replace("/", "_")
    prefix = output_dir / f"head_cycle_ablation_{safe_model}_L{args.layer}"

    raw_df.to_csv(f"{prefix}_raw.csv", index=False)
    aligned_df.to_csv(f"{prefix}_aligned.csv", index=False)
    overall_delta_df.to_csv(f"{prefix}_overall_delta.csv", index=False)
    cycle_delta_df.to_csv(f"{prefix}_cycle_delta.csv", index=False)

    metadata = {
        "model_name": args.model_name,
        "checkpoints": args.checkpoints,
        "layer": args.layer,
        "heads": sorted(ablated_df["head"].unique().tolist()),
        "n_samples": args.n_samples,
        "dataset": args.dataset,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "cycle_sizes": cycle_sizes,
        "output_prefix": str(prefix),
    }
    with open(f"{prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Overall delta rows: {len(overall_delta_df)}")
    print(f"Cycle delta rows: {len(cycle_delta_df)}")
    print(f"Saved under: {output_dir}")


if __name__ == "__main__":
    main()
