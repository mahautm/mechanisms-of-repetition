#!/usr/bin/env python3
"""Run natural and ICL generation on minipile inputs.

This script copies the minipile natural/ICL input pattern used in
parrots/aa_fortu/attention_ablation.py:
- natural: sample prompts from JeanKaddour/minipile
- icl: detect cycles from natural generations, then build ICL inputs from cycles
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from parrots.cycle_detection import detect_cycles

# def detect_cycle(generated_tokens: List[int], min_cycle: int = 5, max_cycle: int = 50, pad: int = 0):
#     """Detect the first repeated cycle in a token sequence."""
#     cycle_length = None
#     cycle = None
#     for l in range(min_cycle, min(max_cycle, len(generated_tokens) // 2)):
#         for j in range(len(generated_tokens) - 2 * l):
#             cycle_candidate = generated_tokens[j : j + l]
#             if all(token == pad for token in cycle_candidate):
#                 continue
#             if cycle_candidate == generated_tokens[j + l : j + 2 * l]:
#                 cycle_length = l
#                 cycle = cycle_candidate
#                 return cycle_length, cycle
#     return None, None

def detect_cycle(generated_tokens: List[int], min_cycle: int = 5, max_cycle: int = 50, pad: int = 0):
    cycle, cycle_size, cycle_count = detect_cycles(generated_tokens)
    if cycle_count > 0:
        return cycle_size, cycle
    return None, None


def load_minipile_texts(n_samples: int, seed: int, prompt_size: int, rank: int = 0, n_ranks: int = 1) -> List[str]:
    """Load minipile texts for the natural condition."""
    pile = load_dataset("JeanKaddour/minipile", split="train")
    pile = pile.shuffle(seed=seed).select(range(n_samples))
    
    # Shard the dataset
    samples_per_rank = (n_samples + n_ranks - 1) // n_ranks
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, n_samples)
    if start_idx >= n_samples:
        return []
    
    pile_shard = pile.select(range(start_idx, end_idx))
    return [sample["text"][:prompt_size] for sample in pile_shard]


def process_outputs(model, inputs, texts: List[str], condition: str, n_tokens: int, gen_kwargs: Dict):
    """Generate and return per-sample cycle annotations."""
    results = []
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    outputs = model.generate(**inputs, max_new_tokens=n_tokens, **gen_kwargs)
    for i, output in enumerate(outputs):
        prompt_len = int(prompt_lengths[i])
        generated_tokens = output[prompt_len:].tolist()
        cycle_length, cycle = detect_cycle(generated_tokens)

        if condition == "natural" or cycle_length is None:
            results.append(
                {
                    "condition": condition,
                    "input": texts[i],
                    "prompt_length": prompt_len,
                    "output_tokens": generated_tokens,
                    "cycle_length": int(cycle_length) if cycle_length is not None else 0,
                    "cycle": cycle if cycle is not None else [],
                    "is_cyclical": cycle_length is not None,
                }
            )
    return results


def build_icl_inputs_from_natural(natural_results: List[Dict], device: torch.device, n_tokens: int):
    """Build ICL prompts from detected natural cycles (same pattern as attention_ablation)."""
    icl_input_ids: List[List[int]] = []
    icl_attention: List[List[int]] = []

    for row in natural_results:
        cycle = row.get("cycle") or []
        if not cycle:
            continue
        pad_len = max(0, n_tokens - len(cycle) * 2)
        ids = [0] * pad_len + cycle * 2
        mask = [0] * pad_len + [1] * (2 * len(cycle))
        icl_input_ids.append(ids)
        icl_attention.append(mask)

    if not icl_input_ids:
        return None

    return {
        "input_ids": torch.tensor(icl_input_ids, device=device),
        "attention_mask": torch.tensor(icl_attention, device=device),
    }


def summarize_results(results: List[Dict], elapsed_seconds: float) -> Dict:
    n = len(results)
    if n == 0:
        return {
            "n_samples": 0,
            "cyclical_rate": 0.0,
            "mean_cycle_length": 0.0,
            "mean_output_tokens": 0.0,
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": 0.0,
        }

    cyclical = [r for r in results if bool(r.get("is_cyclical", False))]
    cycle_lengths = [int(r.get("cycle_length", 0)) for r in cyclical]
    output_lens = [len(r.get("output_tokens", [])) for r in results]
    total_tokens = sum(output_lens)

    return {
        "n_samples": n,
        "cyclical_rate": len(cyclical) / n,
        "mean_cycle_length": float(sum(cycle_lengths) / len(cycle_lengths)) if cycle_lengths else 0.0,
        "mean_output_tokens": float(total_tokens / n),
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": float(total_tokens / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
    }


def batched_process_outputs(
    model,
    tokenizer,
    texts: List[str],
    condition: str,
    n_tokens: int,
    gen_kwargs: Dict,
    batch_size: int,
    device: torch.device,
):
    """Tokenize and generate in small batches to avoid GPU OOM."""
    all_results: List[Dict] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_results = process_outputs(
            model=model,
            inputs=inputs,
            texts=batch_texts,
            condition=condition,
            n_tokens=n_tokens,
            gen_kwargs=gen_kwargs,
        )
        all_results.extend(batch_results)
    return all_results


def batched_process_token_inputs(
    model,
    token_inputs: Dict[str, torch.Tensor],
    n_tokens: int,
    gen_kwargs: Dict,
    batch_size: int,
):
    """Run generation for pre-tokenized ICL inputs in batches."""
    all_results: List[Dict] = []
    total = token_inputs["input_ids"].shape[0]
    for i in range(0, total, batch_size):
        batch_inputs = {
            "input_ids": token_inputs["input_ids"][i : i + batch_size],
            "attention_mask": token_inputs["attention_mask"][i : i + batch_size],
        }
        batch_results = process_outputs(
            model=model,
            inputs=batch_inputs,
            texts=["" for _ in range(batch_inputs["input_ids"].shape[0])],
            condition="natural",
            n_tokens=n_tokens,
            gen_kwargs=gen_kwargs,
        )
        all_results.extend(batch_results)
    return all_results


def run(
    model_name: str,
    output_dir: Path,
    n_samples: int,
    prompt_size: int,
    max_new_tokens: int,
    seed: int,
    top_p: Optional[float],
    batch_size: int,
    use_bnb: bool,
    rank: int = 0,
    n_ranks: int = 1,
    revision: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_bnb:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        device_map="auto" if use_bnb else None,
    )
    if not use_bnb:
        model.to(device)
    model.eval()

    if n_ranks > 1:
        rank_suffix = f"_rank{rank}"
    else:
        rank_suffix = ""

    texts = load_minipile_texts(n_samples=n_samples, seed=seed, prompt_size=prompt_size, rank=rank, n_ranks=n_ranks)
    tokenizer.model_max_length = prompt_size

    gen_kwargs = {}
    if top_p is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = top_p

    natural_t0 = time.perf_counter()
    natural_results = batched_process_outputs(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        condition="natural",
        n_tokens=max_new_tokens,
        gen_kwargs=gen_kwargs,
        batch_size=batch_size,
        device=device,
    )
    natural_elapsed = time.perf_counter() - natural_t0
    icl_inputs = build_icl_inputs_from_natural(natural_results, device=device, n_tokens=max_new_tokens)

    natural_df = pd.DataFrame(natural_results)
    natural_out = output_dir / f"pile_natural_results{rank_suffix}.csv"
    natural_df.to_csv(natural_out, index=False)

    icl_df = pd.DataFrame(columns=["condition", "output_tokens", "cycle_length", "cycle", "is_cyclical"])
    icl_results: List[Dict] = []
    icl_elapsed = 0.0
    if icl_inputs is not None:
        icl_t0 = time.perf_counter()
        icl_results = batched_process_token_inputs(
            model=model,
            token_inputs=icl_inputs,
            n_tokens=max_new_tokens,
            gen_kwargs=gen_kwargs,
            batch_size=batch_size,
        )
        icl_elapsed = time.perf_counter() - icl_t0
        for row in icl_results:
            row["condition"] = "icl"
        icl_df = pd.DataFrame(icl_results)

    icl_out = output_dir / f"pile_icl_results{rank_suffix}.csv"
    icl_df.to_csv(icl_out, index=False)

    summary = {
        "model_name": model_name,
        "revision": revision,
        "top_p": top_p,
        "seed": seed,
        "n_samples_requested": n_samples,
        "prompt_size": prompt_size,
        "max_new_tokens": max_new_tokens,
        "natural": summarize_results(natural_results, natural_elapsed),
        "icl": summarize_results(icl_results, icl_elapsed),
    }
    summary_out = output_dir / f"pile_eval_summary{rank_suffix}.json"
    summary_out.write_text(json.dumps(summary, indent=2))

    print(f"Saved natural results to {natural_out}")
    print(f"Saved icl results to {icl_out}")
    print(f"Saved eval summary to {summary_out}")
    print(f"Natural samples: {len(natural_df)}")
    print(f"ICL samples: {len(icl_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run natural+ICL generation on minipile")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--prompt_size", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_bnb", dest="use_bnb", action="store_true", default=True)
    parser.add_argument("--no_use_bnb", dest="use_bnb", action="store_false")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--n_ranks", type=int, default=1)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(
        model_name=args.model_name,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        prompt_size=args.prompt_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        top_p=args.top_p,
        batch_size=args.batch_size,
        use_bnb=args.use_bnb,
        rank=args.rank,
        n_ranks=args.n_ranks,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()