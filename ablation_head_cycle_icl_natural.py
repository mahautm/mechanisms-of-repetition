#!/usr/bin/env python3
"""
Enhanced ablation with paper-defined Natural vs. ICL repetition conditions.

Both conditions are derived from greedy generations on a Minipile raw set:
1. NATURAL: prompts whose greedy continuations develop a trailing repetition cycle;
    if the cycle starts later, the prompt is extended with the model's prefix output
    up to the first repeated token.
2. ICL: prompts that do not repeat at all under greedy decoding; the 32-token raw
    extract is duplicated to form the final prompt.

This matches the dataset definition described in the paper rather than the older
synthetic prompt construction.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def get_output_projection(attention_module: torch.nn.Module) -> torch.nn.Module:
    """Return the attention module's output projection (e.g. o_proj/dense/c_proj).

    Hooking this submodule's *input* (rather than the attention module's
    output) is required for a true per-head ablation: the output
    projection is a full d_model x d_model mixing matrix, so slicing a
    contiguous chunk of its *output* removes a block of residual-stream
    coordinates that mixes contributions from every head, not one head's
    contribution. Its *input* is still the concatenation of per-head
    values, where a contiguous head_dim-sized slice does correspond to a
    single head.
    """
    for attr in ["o_proj", "out_proj", "c_proj", "dense", "wo"]:
        if hasattr(attention_module, attr):
            candidate = getattr(attention_module, attr)
            if isinstance(candidate, torch.nn.Module):
                return candidate
    raise ValueError("Could not find output projection in attention module")


def infer_num_heads(model, attention_module: torch.nn.Module) -> int:
    for attr in ["num_heads", "n_heads", "num_attention_heads", "n_head"]:
        if hasattr(attention_module, attr):
            value = getattr(attention_module, attr)
            if isinstance(value, int) and value > 0:
                return value
    if hasattr(model.config, "num_attention_heads"):
        return int(model.config.num_attention_heads)
    raise ValueError("Could not infer number of attention heads")


def load_natural_prompts(n_samples: int, seed: int, dataset_name: str) -> List[str]:
    """Load natural text for NATURAL condition."""
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


def load_icl_prompts(n_samples: int, seed: int) -> List[str]:
    """
    Load or generate ICL (In-Context Learning) prompts.
    
    ICL prompts contain examples followed by a query to complete.
    Tests model's ability to follow patterns vs. inherent repetition.
    
    Format: "Example1: input1 → output1. Example2: input2 → output2. Query: input?"
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Simple ICL patterns: number sequences, word lists, etc.
    icl_templates = [
        # Arithmetic sequences
        ["1 2 3 4 5", "2 4 6 8 10", "5 10 15 20 25"],
        # Word lists
        ["apple banana orange", "dog cat bird", "red blue green"],
        # Fibonacci-like
        ["1 1 2 3 5", "1 2 3 5 8", "2 3 5 8 13"],
        # Alphabetic
        ["a b c d e", "b c d e f", "x y z a b"],
        # Repeating patterns (should NOT repeat in ICL)
        ["the the the", "cat cat cat", "run run run"],
    ]
    
    prompts = []
    for _ in range(n_samples):
        template_choice = random.randint(0, len(icl_templates) - 1)
        template = icl_templates[template_choice]
        
        # Pick 2 examples + 1 incomplete query
        ex1_idx = random.randint(0, len(template) - 1)
        ex2_idx = random.randint(0, len(template) - 1)
        while ex2_idx == ex1_idx:
            ex2_idx = random.randint(0, len(template) - 1)
        
        ex1 = template[ex1_idx].split()[:3]  # Truncate to 3 elements
        ex2 = template[ex2_idx].split()[:3]
        query = template[(ex1_idx + ex2_idx) % len(template)].split()[:2]  # Incomplete
        
        # Format: "Ex1: [pat] Ex2: [pat] Query: [start]"
        prompt = f"Ex1: {' '.join(ex1)} Ex2: {' '.join(ex2)} Query: {' '.join(query)}"
        prompts.append(prompt)
    
    return prompts


def load_raw_minipile_texts(n_samples: int, seed: int, dataset_name: str) -> List[str]:
    """Load raw Minipile texts used to build the paper-defined prompt sets."""
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


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device that should receive prompt tensors."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def truncate_to_token_ids(text: str, tokenizer, max_tokens: int) -> List[int]:
    """Tokenize a prompt and keep the first max_tokens tokens without adding specials."""
    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]


def build_paper_condition_prompts(
    model,
    tokenizer,
    raw_texts: List[str],
    raw_prompt_tokens: int,
    generation_tokens: int,
    batch_size: int,
) -> Tuple[List[str], List[str]]:
    """Build Natural and ICL prompts from greedy generations on truncated raw texts."""
    device = get_model_device(model)
    tokenizer.padding_side = "left"

    raw_prompt_ids = [truncate_to_token_ids(text, tokenizer, raw_prompt_tokens) for text in raw_texts]
    raw_prompt_ids = [ids for ids in raw_prompt_ids if len(ids) > 0]

    natural_prompts: List[str] = []
    icl_prompts: List[str] = []
    eos_token_id = tokenizer.eos_token_id

    for batch_start in tqdm(range(0, len(raw_prompt_ids), batch_size), desc="Building paper prompts", leave=False):
        batch_ids = raw_prompt_ids[batch_start : batch_start + batch_size]
        prompt_lengths = [len(ids) for ids in batch_ids]
        max_len = max(prompt_lengths)

        input_ids = torch.tensor(
            [([tokenizer.pad_token_id] * (max_len - len(ids))) + ids for ids in batch_ids],
            device=device,
        )
        attention_mask = torch.tensor(
            [([0] * (max_len - len(ids))) + ([1] * len(ids)) for ids in batch_ids],
            device=device,
        )

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=generation_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        for row_idx, prompt_len in enumerate(prompt_lengths):
            prompt_start = max_len - prompt_len
            prompt_tokens = generated[row_idx, prompt_start : prompt_start + prompt_len].detach().cpu().tolist()
            continuation_tokens = generated[row_idx, max_len:].detach().cpu()

            cycle, cycle_size, cycle_count, cycle_start = detect_cycles(
                continuation_tokens,
                return_index=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            if cycle is not None and cycle_size > 1 and cycle_count > 1:
                if eos_token_id is not None and all(token == eos_token_id for token in cycle):
                    continue
                natural_tokens = prompt_tokens + continuation_tokens[:cycle_start].tolist()
                natural_prompts.append(tokenizer.decode(natural_tokens, skip_special_tokens=True))
            else:
                icl_tokens = prompt_tokens + prompt_tokens
                icl_prompts.append(tokenizer.decode(icl_tokens, skip_special_tokens=True))

    return natural_prompts, icl_prompts


def build_paired_condition_prompts(
    tokenizer,
    raw_texts: List[str],
    raw_prompt_tokens: int,
) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    """Build length-matched, source-paired Natural/ICL prompts (no model-based selection).

    Each raw text with at least 2 * raw_prompt_tokens tokens yields one prompt per condition:
      NATURAL: its first 2k tokens (ordinary running text).
      ICL:     its first k tokens, duplicated (k + k tokens).
    Both prompts have exactly 2k tokens and share the same first k tokens. Selection depends
    only on the tokenizer, so the same prompts are used at every checkpoint.
    Returns (source_indices, natural_ids, icl_ids) with source_indices into raw_texts.
    """
    k = raw_prompt_tokens
    source_indices, natural_ids, icl_ids = [], [], []
    for idx, text in enumerate(raw_texts):
        ids = truncate_to_token_ids(text, tokenizer, 2 * k)
        if len(ids) < 2 * k:
            continue
        source_indices.append(idx)
        natural_ids.append(ids[: 2 * k])
        icl_ids.append(ids[:k] + ids[:k])
    return source_indices, natural_ids, icl_ids


def slice_prompts_for_chunk(prompts: List[str], chunk_index: Optional[int], chunk_count: int) -> Tuple[List[str], int, int]:
    if chunk_index is None or chunk_count <= 1:
        return prompts, 0, len(prompts)

    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError(f"chunk_index={chunk_index} out of range for chunk_count={chunk_count}")

    start = (len(prompts) * chunk_index) // chunk_count
    end = (len(prompts) * (chunk_index + 1)) // chunk_count
    return prompts[start:end], start, end


def run_generation(
    model,
    tokenizer,
    prompts: List[str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> Dict[int, Tuple[List[int], int, float]]:
    """
    Generate from prompts and detect cycles.
    Returns: {prompt_idx: (token_ids, cycle_size, cycle_count)}
    """
    model.eval()

    tokenizer.padding_side = "left"
    results = {}

    # Length-sorted batches under a token budget of batch_size * (shortest-case sequence),
    # so long prompts get smaller batches instead of running out of GPU memory. With
    # equal-length prompts this reduces to plain fixed-size batches in the original order.
    lengths = [
        min(len(tokenizer(p, add_special_tokens=True)["input_ids"]) if isinstance(p, str) else len(p), max_prompt_tokens)
        for p in prompts
    ]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    token_budget = batch_size * (min(lengths, default=0) + max_new_tokens)
    batches, current = [], []
    for i in order:
        width = lengths[i] + max_new_tokens
        if current and (len(current) >= batch_size or (len(current) + 1) * width > token_budget):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)

    with torch.no_grad():
        for batch_indices in tqdm(batches, desc="Generating sequences", leave=False):
            batch_prompts = [prompts[i] for i in batch_indices]

            if isinstance(batch_prompts[0], str):
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_tokens,
                ).to(model.device)
            else:
                # Pre-tokenized prompts (paired mode): left-pad token ids directly so the
                # prompt length is exactly what was built, with no decode/re-encode drift.
                batch_ids = [list(ids[:max_prompt_tokens]) for ids in batch_prompts]
                max_len = max(len(ids) for ids in batch_ids)
                encoded = {
                    "input_ids": torch.tensor(
                        [[tokenizer.pad_token_id] * (max_len - len(ids)) + ids for ids in batch_ids],
                        device=model.device,
                    ),
                    "attention_mask": torch.tensor(
                        [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in batch_ids],
                        device=model.device,
                    ),
                }

            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Left padding: every row's continuation starts at the padded input width.
            # (Slicing at the unpadded prompt length would prepend the tail of the prompt.)
            input_width = encoded["input_ids"].shape[1]
            for j, prompt_idx in enumerate(batch_indices):
                gen_token_ids = generated[j, input_width:].cpu().tolist()

                cycle, cycle_size, cycle_count = detect_cycles(gen_token_ids)

                if cycle is not None and cycle_size > 0:
                    results[prompt_idx] = (
                        gen_token_ids,
                        cycle_size,
                        cycle_count,
                    )
                else:
                    results[prompt_idx] = (gen_token_ids, 0, 0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def create_head_ablation_hook(layer_idx: int, head_idx: int, num_heads: int):
    """Create a pre-hook on the output projection that isolates one head.

    Registered as a forward-pre-hook on the attention module's output
    projection (o_proj/dense/c_proj/...), so it sees the per-head
    concatenation *before* it gets mixed across heads by that projection.
    A contiguous head_dim-sized slice of this input corresponds to a real
    head, unlike a slice of the projection's output.
    """

    def hook_fn(module, args):
        if not args:
            return args
        hidden_states = args[0]

        # hidden_states shape: (batch_size, seq_len, hidden_dim)
        # Each head contributes hidden_dim // num_heads dimensions
        head_dim = hidden_states.size(-1) // num_heads
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim

        hidden_states = hidden_states.clone()
        hidden_states[:, :, start:end] = 0.0

        return (hidden_states, *args[1:])

    return hook_fn


def compute_repetition_metrics(cycle_results: Dict) -> Tuple[float, float]:
    """
    Compute repetition_rate and avg_cycle_count from cycle detection results.
    
    cycle_results: {prompt_idx: (token_ids, cycle_size, cycle_count)}
    """
    n_prompts = len(cycle_results)
    # Require true repeated cycles, not just a single terminal block.
    n_repeating = sum(1 for _, _, count in cycle_results.values() if count > 1)
    repetition_rate = n_repeating / n_prompts if n_prompts > 0 else 0.0
    avg_cycle_count = sum(count for _, _, count in cycle_results.values()) / n_prompts if n_prompts > 0 else 0.0
    return repetition_rate, avg_cycle_count


def per_prompt_records(cycle_results: Dict, prompt_ids: List[int], eos_token_id: Optional[int], base: Dict) -> List[Dict]:
    """One row per prompt, for paired (McNemar) tests of baseline vs. ablated outcomes.

    repeats: the paper metric used in compute_repetition_metrics (cycle_count > 1).
    repeats_no_eos: same, after stripping trailing EOS/pad tokens, so a generation that
        simply stopped (EOS, then EOS padding) is not counted as a repetition cycle.
    """
    rows = []
    for local_idx, (token_ids, cycle_size, cycle_count) in sorted(cycle_results.items()):
        stripped = list(token_ids)
        eos_terminated = False
        while eos_token_id is not None and stripped and stripped[-1] == eos_token_id:
            stripped.pop()
            eos_terminated = True
        if eos_terminated:
            _, s_size, s_count = detect_cycles(stripped)
        else:
            s_size, s_count = cycle_size, cycle_count
        rows.append({
            **base,
            "prompt_idx": prompt_ids[local_idx],
            "repeats": int(cycle_count > 1),
            "cycle_size": cycle_size,
            "cycle_count": cycle_count,
            "eos_terminated": int(eos_terminated),
            "repeats_no_eos": int(s_count > 1),
        })
    return rows


def _revision_exists(model_name: str, revision: str) -> bool:
    """Check via the Hub API whether a revision genuinely exists for a model.

    Used to decide the step0 fallback. We deliberately do NOT infer "revision
    missing" from a from_pretrained() failure: under concurrent load, a brand
    new revision being downloaded by many workers at once can produce
    transient cache-corruption errors (truncated safetensors headers,
    partially-written config.json) that look identical to a missing
    revision but aren't — silently falling back to a random-init model in
    that case would mislabel corrupted-load results as genuine step0 data.
    """
    from huggingface_hub import list_repo_refs

    try:
        refs = list_repo_refs(model_name)
        names = {b.name for b in refs.branches} | {t.name for t in refs.tags}
        return revision in names
    except Exception:
        # Can't determine (offline, etc.) — don't guess; let the normal load
        # path run and fail loudly if the revision truly can't be fetched.
        return True


def build_untrained_model_and_tokenizer(model_name: str, use_bnb: bool = False, seed: int = 0):
    """Build a freshly-initialized (random-weight) model as a step0 proxy.

    Used when a model family doesn't publish an actual "step0" revision
    (the saved initialization before any optimizer step) — this is the best
    available stand-in for "before any training".
    """
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(torch.float16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def load_model_and_tokenizer(model_name: str, checkpoint: str, use_bnb: bool = False):
    """Load model/tokenizer with retry on transient safetensors header errors."""
    if checkpoint.startswith("randinit_seed"):
        # Independent random initialisation (HF from_config default init), seed in the name.
        seed = int(checkpoint[len("randinit_seed"):])
        print(f"[INFO] Building freshly-initialized {model_name} with seed={seed}")
        return build_untrained_model_and_tokenizer(model_name, use_bnb=use_bnb, seed=seed)
    if checkpoint == "step0" and not _revision_exists(model_name, "step0"):
        print(
            f"[INFO] {model_name} has no 'step0' revision; using a "
            "freshly-initialized (random-weight) model as the pre-training proxy."
        )
        return build_untrained_model_and_tokenizer(model_name, use_bnb=use_bnb)
    return _load_pretrained_checkpoint(model_name, checkpoint, use_bnb)


def _load_pretrained_checkpoint(model_name: str, checkpoint: str, use_bnb: bool = False):
    base_model_kwargs = {"trust_remote_code": True}
    if use_bnb:
        base_model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model_kwargs["device_map"] = "auto"
    else:
        base_model_kwargs["torch_dtype"] = torch.float16
    base_tok_kwargs = {"trust_remote_code": True}

    if checkpoint != "steplatest":
        base_model_kwargs["revision"] = checkpoint
        base_tok_kwargs["revision"] = checkpoint

    attempts = [
        {"force_download": False, "use_safetensors": True},
        {"force_download": True, "use_safetensors": True},
        {"force_download": True, "use_safetensors": False},
    ]

    last_error = None
    for idx, opts in enumerate(attempts, start=1):
        try:
            model_kwargs = dict(base_model_kwargs)
            model_kwargs.update(opts)

            tok_kwargs = dict(base_tok_kwargs)
            tok_kwargs["force_download"] = opts["force_download"]

            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            return model, tokenizer
        except Exception as exc:
            last_error = exc
            # Retry on any load failure, not just specific message patterns:
            # concurrent first-time downloads of the same revision can corrupt
            # the shared HF cache in many different ways (truncated safetensors
            # headers, partially-written config.json/pytorch_model.bin), and
            # pattern-matching on message text proved to miss real cases
            # (e.g. "Error while deserializing header: MetadataIncompleteBuffer"
            # doesn't contain "safetensors"). force_download on retry re-fetches
            # a clean copy.
            if idx == len(attempts):
                raise

            print(
                f"[WARN] Load failed at checkpoint '{checkpoint}' (attempt {idx}/{len(attempts)}): {exc}"
            )
            print("[WARN] Retrying with stronger download fallback...")
            time.sleep(2)

    raise last_error


def main():
    parser = argparse.ArgumentParser(
        description="Head ablation with ICL vs. Natural conditions"
    )
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=["step0", "step1", "step1000", "step5000", "step10000", "step100000", "steplatest"],
    )
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--heads", type=str, default="", help="Comma-separated head indices")
    parser.add_argument("--n_samples", type=int, default=1000, help="Max prompts to use per condition after classification")
    parser.add_argument("--raw_pool_size", type=int, default=1000, help="Raw Minipile prompts to classify per checkpoint")
    parser.add_argument("--raw_prompt_tokens", type=int, default=32, help="Token length of the raw sentence extract")
    parser.add_argument("--prompt_generation_tokens", type=int, default=1000, help="Greedy generation length used to classify raw prompts")
    parser.add_argument("--natural_dataset", type=str, default="JeanKaddour/minipile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--chunk_index", type=int, default=None, help="Optional prompt shard index")
    parser.add_argument("--chunk_count", type=int, default=1, help="Total number of prompt shards")
    parser.add_argument("--output_dir", type=str, default="./outputs_ablation_head_cycle")
    parser.add_argument("--use_bnb", action="store_true", help="Use 4-bit bitsandbytes loading")
    parser.add_argument(
        "--prompt_mode",
        choices=["selected", "paired"],
        default="selected",
        help="selected: paper construction (per-checkpoint behavioural selection). "
        "paired: every raw text yields a length-matched Natural and ICL prompt, identical across checkpoints.",
    )
    parser.add_argument(
        "--baseline_replicate",
        action="store_true",
        help="Run the unablated baseline twice (head=-2) to measure the GPU-nondeterminism noise floor.",
    )
    parser.add_argument("--baseline_only", action="store_true", help="Skip head ablations (baseline repetition survey only)")

    args = parser.parse_args()
    set_seed(args.seed)

    # Load raw prompts once, then rebuild the paper-defined conditions per checkpoint.
    print("Loading raw Minipile prompts...")
    raw_texts = load_raw_minipile_texts(args.raw_pool_size, args.seed, args.natural_dataset)

    paired = None
    if args.prompt_mode == "paired":
        # Prompts depend only on the tokenizer, so build them once, before any checkpoint.
        pair_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        source_idx, nat_ids, icl_ids = build_paired_condition_prompts(pair_tokenizer, raw_texts, args.raw_prompt_tokens)
        source_idx, nat_ids, icl_ids = source_idx[: args.n_samples], nat_ids[: args.n_samples], icl_ids[: args.n_samples]
        print(f"  ✓ Paired sources with >= {2 * args.raw_prompt_tokens} tokens: {len(source_idx)} (of {len(raw_texts)} raw)")
        source_idx, raw_start, raw_end = slice_prompts_for_chunk(source_idx, args.chunk_index, args.chunk_count)
        paired = (source_idx, nat_ids[raw_start:raw_end], icl_ids[raw_start:raw_end])
    else:
        raw_texts, raw_start, raw_end = slice_prompts_for_chunk(raw_texts, args.chunk_index, args.chunk_count)
    print(f"  ✓ Prompts in this shard: {raw_end - raw_start}")
    if args.chunk_index is not None and args.chunk_count > 1:
        print(f"  ✓ Prompt shard: {args.chunk_index + 1}/{args.chunk_count} ({raw_start}:{raw_end})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    prompt_rows = []

    for checkpoint in args.checkpoints:
        print(f"\n{'='*80}")
        print(f"Checkpoint: {checkpoint}")
        print("=" * 80)

        # Load model
        model, tokenizer = load_model_and_tokenizer(args.model_name, checkpoint, args.use_bnb)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not args.use_bnb:
            model.to(device)

        if paired is not None:
            natural_ids_for_rows, natural_prompts, icl_prompts = paired
            icl_ids_for_rows = natural_ids_for_rows
        else:
            print("\n  Building condition prompts from greedy generations...")
            natural_prompts, icl_prompts = build_paper_condition_prompts(
                model=model,
                tokenizer=tokenizer,
                raw_texts=raw_texts,
                raw_prompt_tokens=args.raw_prompt_tokens,
                generation_tokens=args.prompt_generation_tokens,
                batch_size=args.batch_size,
            )

            natural_prompts = natural_prompts[: args.n_samples]
            icl_prompts = icl_prompts[: args.n_samples]
            # Selected mode: prompts are not paired across conditions; index within this shard.
            natural_ids_for_rows = list(range(len(natural_prompts)))
            icl_ids_for_rows = list(range(len(icl_prompts)))

        print(f"  ✓ Natural: {len(natural_prompts)} prompts after filtering")
        print(f"  ✓ ICL: {len(icl_prompts)} prompts after filtering")

        layers = get_transformer_layers(model)
        if args.layer >= len(layers):
            print(f"❌ Layer {args.layer} out of range (model has {len(layers)} layers)")
            continue

        layer = layers[args.layer]
        attn_module = get_attention_module(layer)
        out_proj = get_output_projection(attn_module)
        num_heads = infer_num_heads(model, attn_module)

        head_indices = [] if args.baseline_only else (parse_int_list(args.heads) or list(range(num_heads)))
        eos_id = tokenizer.eos_token_id

        # Run for both conditions
        for condition_name, prompts, row_ids in [
            ("natural", natural_prompts, natural_ids_for_rows),
            ("icl", icl_prompts, icl_ids_for_rows),
        ]:
            print(f"\n  Condition: {condition_name.upper()}")
            if not prompts:
                print("    (no prompts)")
                continue
            row_base = {"checkpoint": checkpoint, "layer": args.layer, "condition": condition_name, "chunk_index": args.chunk_index}

            # Baseline (no ablation)
            print(f"    Baseline...", end=" ", flush=True)
            baseline_results = run_generation(
                model,
                tokenizer,
                prompts,
                args.max_prompt_tokens,
                args.max_new_tokens,
                args.do_sample,
                args.temperature,
                args.top_p,
                args.batch_size,
            )
            baseline_rep_rate, baseline_cycle_count = compute_repetition_metrics(baseline_results)
            print(f"rep_rate={baseline_rep_rate:.3f}, cycle_count={baseline_cycle_count:.2f}")
            prompt_rows.extend(per_prompt_records(baseline_results, row_ids, eos_id, {**row_base, "head": -1}))

            if args.baseline_replicate:
                replicate_results = run_generation(
                    model,
                    tokenizer,
                    prompts,
                    args.max_prompt_tokens,
                    args.max_new_tokens,
                    args.do_sample,
                    args.temperature,
                    args.top_p,
                    args.batch_size,
                )
                prompt_rows.extend(per_prompt_records(replicate_results, row_ids, eos_id, {**row_base, "head": -2}))

            for head_idx in tqdm(head_indices, desc=f"    Heads ({condition_name})", leave=False):
                # Ablate this head
                hook = create_head_ablation_hook(args.layer, head_idx, num_heads)
                handle = out_proj.register_forward_pre_hook(hook)

                try:
                    ablated_results = run_generation(
                        model,
                        tokenizer,
                        prompts,
                        args.max_prompt_tokens,
                        args.max_new_tokens,
                        args.do_sample,
                        args.temperature,
                        args.top_p,
                        args.batch_size,
                    )
                    ablated_rep_rate, ablated_cycle_count = compute_repetition_metrics(ablated_results)
                    prompt_rows.extend(per_prompt_records(ablated_results, row_ids, eos_id, {**row_base, "head": head_idx}))

                    all_rows.append({
                        "checkpoint": checkpoint,
                        "layer": args.layer,
                        "head": head_idx,
                        "condition": condition_name,
                        "n_prompts": len(prompts),
                        "baseline_repetition_rate": baseline_rep_rate,
                        "baseline_avg_cycle_count": baseline_cycle_count,
                        "ablated_repetition_rate": ablated_rep_rate,
                        "ablated_avg_cycle_count": ablated_cycle_count,
                        "delta_repetition_rate": ablated_rep_rate - baseline_rep_rate,
                        "delta_avg_cycle_count": ablated_cycle_count - baseline_cycle_count,
                    })
                finally:
                    handle.remove()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(all_rows)
    safe_model = args.model_name.replace("/", "_")
    output_file = output_dir / f"head_cycle_ablation_{safe_model}_L{args.layer}_icl_natural.csv"
    df.to_csv(output_file, index=False)
    prompt_file = output_dir / f"head_cycle_ablation_{safe_model}_L{args.layer}_per_prompt.csv.gz"
    pd.DataFrame(prompt_rows).to_csv(prompt_file, index=False)
    print(f"✓ Saved per-prompt outcomes: {prompt_file} ({len(prompt_rows)} rows)")

    print("\n" + "=" * 80)
    print(f"✓ Saved: {output_file}")
    print(f"  Total rows: {len(df)}")
    if len(df):
        print(f"  Natural rows: {len(df[df['condition']=='natural'])}")
        print(f"  ICL rows: {len(df[df['condition']=='icl'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
