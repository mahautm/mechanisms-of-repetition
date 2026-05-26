#!/usr/bin/env python3
"""Sanity probe for Apertus long greedy decoding baselines."""

from ablation_head_cycle_icl_natural import (
    compute_repetition_metrics,
    load_icl_prompts,
    load_model_and_tokenizer,
    load_natural_prompts,
    run_generation,
    set_seed,
)
import torch


def main() -> None:
    model_name = "swiss-ai/Apertus-8B-2509"
    checkpoints = [
        "step50000-tokens210B",
        "step650000-tokens2730B",
        "step1432000-tokens6014B",
        "step2627139-tokens15T",
    ]

    n_samples = 32
    max_prompt_tokens = 64
    max_new_tokens = 512
    batch_size = 16
    seed = 42

    set_seed(seed)
    natural = load_natural_prompts(n_samples, seed, "JeanKaddour/minipile")
    icl = load_icl_prompts(n_samples, seed)

    print(
        "[CONFIG] "
        f"n_samples={n_samples} max_prompt_tokens={max_prompt_tokens} "
        f"max_new_tokens={max_new_tokens} batch_size={batch_size} do_sample=False",
        flush=True,
    )

    for ckpt in checkpoints:
        print(f"\n[CHECKPOINT] {ckpt}", flush=True)
        model, tok = load_model_and_tokenizer(model_name, ckpt, use_bnb=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        nat = run_generation(
            model,
            tok,
            natural,
            max_prompt_tokens,
            max_new_tokens,
            False,
            0.8,
            0.9,
            batch_size,
        )
        nat_rep, nat_cycle = compute_repetition_metrics(nat)
        print(
            f"NATURAL baseline_repetition_rate={nat_rep:.6f} "
            f"baseline_avg_cycle_count={nat_cycle:.6f}",
            flush=True,
        )

        icl_res = run_generation(
            model,
            tok,
            icl,
            max_prompt_tokens,
            max_new_tokens,
            False,
            0.8,
            0.9,
            batch_size,
        )
        icl_rep, icl_cycle = compute_repetition_metrics(icl_res)
        print(
            f"ICL baseline_repetition_rate={icl_rep:.6f} "
            f"baseline_avg_cycle_count={icl_cycle:.6f}",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[DONE] sanity probe complete", flush=True)


if __name__ == "__main__":
    main()
