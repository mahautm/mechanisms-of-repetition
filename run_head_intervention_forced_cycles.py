#!/usr/bin/env python3
"""Evaluate selective head policies vs top-p for natural repetition reduction with ICL safety."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    is_cyclical: bool
    cycle_length: int
    cycle_count: int
    onset_position: int
    cycle_tightness: float
    cycle_delay: int
    cycle_consistency: int

    def as_dict(self) -> Dict:
        return {
            "is_cyclical": self.is_cyclical,
            "cycle_length": self.cycle_length,
            "cycle_count": self.cycle_count,
            "onset_position": self.onset_position,
            "cycle_tightness": self.cycle_tightness,
            "cycle_delay": self.cycle_delay,
            "cycle_consistency": self.cycle_consistency,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_cycle_rich(tokens: List[int], min_cycle: int = 5, max_cycle: int = 50, pad: int = 0):
    """Detect first repeated cycle and return rich cycle metadata."""
    n = len(tokens)
    for length in range(min_cycle, min(max_cycle, n // 2) + 1):
        for start in range(0, n - 2 * length + 1):
            candidate = tokens[start : start + length]
            if all(t == pad for t in candidate):
                continue

            repeat_count = 0
            idx = start + length
            while idx + length <= n and tokens[idx : idx + length] == candidate:
                repeat_count += 1
                idx += length

            if repeat_count >= 1:
                full_count = repeat_count + 1
                total_cycled = full_count * length
                return {
                    "cycle": candidate,
                    "cycle_length": length,
                    "cycle_count": full_count,
                    "onset_position": start,
                    "cycle_tightness": total_cycled / max(n, 1),
                    "cycle_delay": start,
                    "cycle_consistency": repeat_count,
                }
    return None


def compute_cycle_metrics(tokens: List[int]) -> CycleMetrics:
    info = detect_cycle_rich(tokens)
    if info is None:
        return CycleMetrics(False, 0, 0, 0, 0.0, 0, 0)
    return CycleMetrics(
        True,
        int(info["cycle_length"]),
        int(info["cycle_count"]),
        int(info["onset_position"]),
        float(info["cycle_tightness"]),
        int(info["cycle_delay"]),
        int(info["cycle_consistency"]),
    )


def load_minipile_texts(n_samples: int, seed: int, prompt_size: int) -> List[str]:
    pile = load_dataset("JeanKaddour/minipile", split="train")
    pile = pile.shuffle(seed=seed).select(range(min(n_samples, len(pile))))
    return [sample["text"][:prompt_size] for sample in pile]


def build_icl_prompts(n_samples: int, seed: int) -> List[str]:
    random.seed(seed)
    templates = [
        ["1 2 3 4 5", "2 4 6 8 10", "3 6 9 12 15"],
        ["apple banana orange", "dog cat bird", "red blue green"],
        ["a b c d e", "b c d e f", "x y z a b"],
        ["north south east", "winter spring summer", "monday tuesday wednesday"],
    ]
    prompts: List[str] = []
    for _ in range(n_samples):
        t = random.choice(templates)
        ex1 = random.choice(t).split()[:3]
        ex2 = random.choice(t).split()[:3]
        q = random.choice(t).split()[:2]
        prompts.append(f"Ex1: {' '.join(ex1)} Ex2: {' '.join(ex2)} Query: {' '.join(q)}")
    return prompts


def get_transformer_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError("Unsupported model architecture")


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attn"):
        return layer.attn
    raise ValueError("Could not find attention module")


def infer_num_heads(model, attention_module: torch.nn.Module) -> int:
    for attr in ["num_heads", "n_heads", "num_attention_heads", "n_head"]:
        if hasattr(attention_module, attr):
            v = getattr(attention_module, attr)
            if isinstance(v, int) and v > 0:
                return v
    if hasattr(model.config, "num_attention_heads"):
        return int(model.config.num_attention_heads)
    raise ValueError("Could not infer number of heads")


def load_model_and_tokenizer(model_name: str, use_bnb: bool):
    """Load a causal LM, falling back to standard loading if bitsandbytes is unavailable."""
    if use_bnb:
        try:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            return model, tokenizer, True
        except Exception as exc:
            logger.warning("4-bit loading failed for %s; falling back to standard loading: %s", model_name, exc)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer, False
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        logger.warning("GPU fp16 loading OOM for %s; retrying on CPU in float32", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer, False


def create_head_hook(head_idx: int, num_heads: int, mode: str, factor: float):
    def hook_fn(module, inputs, output):
        tuple_payload = None
        if isinstance(output, tuple):
            if not output:
                return output
            attn_out = output[0]
            tuple_payload = output[1:]
        else:
            attn_out = output

        head_dim = attn_out.size(-1) // num_heads
        s = head_idx * head_dim
        e = (head_idx + 1) * head_dim

        attn_out = attn_out.clone()
        if mode == "amplify":
            attn_out[:, :, s:e] = attn_out[:, :, s:e] * factor
        elif mode == "suppress":
            attn_out[:, :, s:e] = 0.0

        if tuple_payload is not None:
            return (attn_out, *tuple_payload)
        return attn_out

    return hook_fn


def register_policy_hooks(model, interventions: List[Dict]) -> List[torch.utils.hooks.RemovableHandle]:
    handles = []
    layers = get_transformer_layers(model)

    for spec in interventions:
        layer_idx = int(spec["layer"])
        head_idx = int(spec["head"])
        mode = str(spec["mode"])
        factor = float(spec.get("factor", 1.0))

        if layer_idx < 0 or layer_idx >= len(layers):
            logger.warning("Skipping out-of-range layer %s", layer_idx)
            continue
        attn_mod = get_attention_module(layers[layer_idx])
        n_heads = infer_num_heads(model, attn_mod)
        if head_idx < 0 or head_idx >= n_heads:
            logger.warning("Skipping invalid head L%sH%s (n_heads=%s)", layer_idx, head_idx, n_heads)
            continue

        hook = create_head_hook(head_idx, n_heads, mode=mode, factor=factor)
        handles.append(attn_mod.register_forward_hook(hook))

    return handles


def run_generation(
    model,
    tokenizer,
    prompts: List[str],
    condition: str,
    arm: str,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    gen_kwargs: Dict,
    interventions: Optional[List[Dict]] = None,
) -> List[Dict]:
    results: List[Dict] = []
    tokenizer.padding_side = "left"

    handles = []
    if interventions:
        handles = register_policy_hooks(model, interventions)

    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            ).to(model.device)

            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            call_kwargs = dict(gen_kwargs)
            # Sampling on some models can emit invalid probabilities; these flags
            # keep decoding numerically stable while preserving top-p behavior.
            if call_kwargs.get("do_sample", False):
                call_kwargs.setdefault("remove_invalid_values", True)
                call_kwargs.setdefault("renormalize_logits", True)

            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    **call_kwargs,
                )

            for i, out in enumerate(outputs):
                p_len = int(prompt_lengths[i])
                gen_tokens = out[p_len:].tolist()
                m = compute_cycle_metrics(gen_tokens)
                results.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "input": batch_prompts[i],
                        "prompt_length": p_len,
                        "output_tokens": gen_tokens,
                        **m.as_dict(),
                    }
                )
    finally:
        for h in handles:
            h.remove()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                # If CUDA is in an asserted state, cache cleanup can also fail.
                pass

    return results


def summarize(df: pd.DataFrame, delta_nat_threshold: float, eps_icl: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        df.groupby(["arm", "condition"], as_index=False)
        .agg(
            n=("is_cyclical", "size"),
            cyclical_rate=("is_cyclical", "mean"),
            mean_cycle_length=("cycle_length", "mean"),
            mean_onset=("onset_position", "mean"),
            mean_tightness=("cycle_tightness", "mean"),
            mean_delay=("cycle_delay", "mean"),
            mean_consistency=("cycle_consistency", "mean"),
        )
    )

    baseline_nat = float(grouped[(grouped["arm"] == "baseline") & (grouped["condition"] == "natural")]["cyclical_rate"].iloc[0])
    baseline_icl = float(grouped[(grouped["arm"] == "baseline") & (grouped["condition"] == "icl")]["cyclical_rate"].iloc[0])

    rows = []
    for arm in sorted(grouped["arm"].unique()):
        nat_row = grouped[(grouped["arm"] == arm) & (grouped["condition"] == "natural")]
        icl_row = grouped[(grouped["arm"] == arm) & (grouped["condition"] == "icl")]
        if nat_row.empty or icl_row.empty:
            continue

        nat = float(nat_row["cyclical_rate"].iloc[0])
        icl = float(icl_row["cyclical_rate"].iloc[0])
        nat_reduction_abs = baseline_nat - nat
        nat_reduction_rel = nat_reduction_abs / max(baseline_nat, 1e-9)
        icl_change_abs = abs(icl - baseline_icl)

        rows.append(
            {
                "arm": arm,
                "baseline_nat_cyclical_rate": baseline_nat,
                "baseline_icl_cyclical_rate": baseline_icl,
                "nat_cyclical_rate": nat,
                "icl_cyclical_rate": icl,
                "nat_reduction_abs": nat_reduction_abs,
                "nat_reduction_rel": nat_reduction_rel,
                "icl_change_abs": icl_change_abs,
                "meets_nat_target": nat_reduction_rel >= delta_nat_threshold,
                "meets_icl_safety": icl_change_abs <= eps_icl,
                "pass_noninferiority": (nat_reduction_rel >= delta_nat_threshold) and (icl_change_abs <= eps_icl),
            }
        )

    return grouped.sort_values(["arm", "condition"]), pd.DataFrame(rows).sort_values("arm")


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a compact markdown table without optional dependencies."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selective head policy evaluation")
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-1B-hf")
    parser.add_argument("--policy_json", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delta_nat_threshold", type=float, default=0.20)
    parser.add_argument("--epsilon_icl", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default="outputs/head_policy_eval")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.policy_json, "r") as f:
        policy = json.load(f)
    interventions = policy.get("interventions", [])
    if not interventions:
        raise ValueError("Policy has no interventions")

    logger.info("Loading model %s", args.model_name)
    model, tokenizer, used_bnb = load_model_and_tokenizer(args.model_name, args.use_bnb)
    logger.info("Model loading mode: %s", "4-bit" if used_bnb else "standard fp16")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    natural_prompts = load_minipile_texts(args.n_samples, args.seed, prompt_size=512)
    icl_prompts = build_icl_prompts(args.n_samples, args.seed)

    arms = [
        ("baseline", {"do_sample": False, "top_p": 1.0}, None),
        ("top_p_0.5", {"do_sample": True, "top_p": 0.5, "temperature": args.temperature}, None),
        ("top_p_0.9", {"do_sample": True, "top_p": 0.9, "temperature": args.temperature}, None),
        ("head_policy", {"do_sample": False, "top_p": 1.0}, interventions),
        ("hybrid_top_p_0.9", {"do_sample": True, "top_p": 0.9, "temperature": args.temperature}, interventions),
    ]

    all_rows: List[Dict] = []
    t0 = time.time()
    for arm_name, gen_kwargs, policy_interventions in arms:
        logger.info("Running arm: %s", arm_name)

        nat_rows = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompts=natural_prompts,
            condition="natural",
            arm=arm_name,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            gen_kwargs=gen_kwargs,
            interventions=policy_interventions,
        )
        all_rows.extend(nat_rows)

        icl_rows = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompts=icl_prompts,
            condition="icl",
            arm=arm_name,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            gen_kwargs=gen_kwargs,
            interventions=policy_interventions,
        )
        all_rows.extend(icl_rows)

    elapsed = time.time() - t0
    logger.info("Finished evaluation in %.1fs", elapsed)

    df = pd.DataFrame(all_rows)
    safe_model = args.model_name.replace("/", "_")
    raw_json = out_dir / f"policy_eval_{safe_model}_raw.json"
    raw_csv = out_dir / f"policy_eval_{safe_model}_raw.csv"
    summary_csv = out_dir / f"policy_eval_{safe_model}_summary.csv"
    noninf_csv = out_dir / f"policy_eval_{safe_model}_noninferiority.csv"

    with open(raw_json, "w") as f:
        json.dump(all_rows, f)
    df.to_csv(raw_csv, index=False)

    summary_df, noninf_df = summarize(df, args.delta_nat_threshold, args.epsilon_icl)
    summary_df.to_csv(summary_csv, index=False)
    noninf_df.to_csv(noninf_csv, index=False)

    md_path = out_dir / f"policy_eval_{safe_model}_report.md"
    lines = [
        "# Selective Head Policy Evaluation",
        "",
        f"- Model: {args.model_name}",
        f"- n_samples: {args.n_samples}",
        f"- delta_nat_threshold: {args.delta_nat_threshold}",
        f"- epsilon_icl: {args.epsilon_icl}",
        f"- elapsed_seconds: {elapsed:.1f}",
        "",
        "## Non-Inferiority Table",
        "",
    ]
    if noninf_df.empty:
        lines.append("No rows produced.")
    else:
        lines.append(df_to_markdown(noninf_df))
    md_path.write_text("\n".join(lines))

    logger.info("Saved: %s", raw_csv)
    logger.info("Saved: %s", summary_csv)
    logger.info("Saved: %s", noninf_csv)
    logger.info("Saved: %s", md_path)


if __name__ == "__main__":
    main()
