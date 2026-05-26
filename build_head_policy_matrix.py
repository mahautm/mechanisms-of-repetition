#!/usr/bin/env python3
"""Build a broad model/hypothesis matrix for selective head policy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", type=str, default="outputs/head_policy_matrix/matrix.csv")
    parser.add_argument("--policy_root", type=str, default="outputs/head_policy_library")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    args = parser.parse_args()

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    policy_root = Path(args.policy_root)

    models = [
        ("pythia", "70m", "EleutherAI/pythia-70m", "pythia70m"),
        ("pythia", "1.4b", "EleutherAI/pythia-1.4b", "pythia14b"),
        ("pythia", "2.8b", "EleutherAI/pythia-2.8b", "pythia14b_transfer"),
        ("olmo", "1b", "allenai/OLMo-1B-hf", "olmo1b"),
        ("olmo", "1b-0724", "allenai/OLMo-1B-0724-hf", "olmo1b_transfer"),
        ("apertus", "8b", "swiss-ai/Apertus-8B-2509", "apertus_transfer"),
    ]

    hypotheses = [
        ("h1_selective_balanced", "policy_lambda1.0_k2_factor1.25.json"),
        ("h2_selective_conservative", "policy_lambda2.0_k2_factor1.10.json"),
        ("h3_selective_aggressive", "policy_lambda0.5_k4_factor1.50.json"),
    ]

    rows = []
    for family, size, model_name, policy_group in models:
        for hyp_name, policy_name in hypotheses:
            rows.append(
                {
                    "family": family,
                    "size": size,
                    "model_name": model_name,
                    "hypothesis": hyp_name,
                    "policy_group": policy_group,
                    "policy_json": str(policy_root / policy_group / policy_name),
                    "n_samples": args.n_samples,
                    "batch_size": 2,
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "use_bnb": 1,
                    "delta_nat_threshold": 0.20,
                    "epsilon_icl": 0.02,
                    "status": "pending",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved matrix ({len(df)} rows): {out_csv}")


if __name__ == "__main__":
    main()
