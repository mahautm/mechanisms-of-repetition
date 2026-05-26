#!/usr/bin/env python3
"""Build selective head policies from condition-aware ablation outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def checkpoint_sort_key(checkpoint: str):
    match = re.search(r"step(\d+)", str(checkpoint))
    if match:
        return int(match.group(1)), str(checkpoint)
    if str(checkpoint) == "steplatest":
        return 10**12, str(checkpoint)
    return 10**12 - 1, str(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score selective heads for natural-vs-ICL control")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/head_policy")
    parser.add_argument("--lambda_collateral", type=float, default=1.0)
    parser.add_argument("--top_k_amplify", type=int, default=2)
    parser.add_argument("--top_k_suppress", type=int, default=2)
    parser.add_argument("--latest_only", action="store_true")
    parser.add_argument("--default_amplify_factor", type=float, default=1.25)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = {"checkpoint", "layer", "head", "condition", "delta_repetition_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.latest_only:
        checkpoints = sorted(df["checkpoint"].astype(str).unique().tolist(), key=checkpoint_sort_key)
        latest = checkpoints[-1]
        df = df[df["checkpoint"].astype(str) == latest].copy()

    grouped = (
        df.groupby(["layer", "head", "condition"], as_index=False)
        .agg(delta_rep=("delta_repetition_rate", "mean"))
    )

    pivot = grouped.pivot_table(
        index=["layer", "head"],
        columns="condition",
        values="delta_rep",
        aggfunc="mean",
    ).reset_index()

    if "natural" not in pivot.columns or "icl" not in pivot.columns:
        raise ValueError("Input must contain both natural and icl rows")

    pivot["B_natural"] = pivot["natural"]
    pivot["C_icl_abs"] = pivot["icl"].abs()

    # Amplify heads that are natural suppressors in intact model (ablation delta > 0).
    pivot["amplify_benefit"] = np.where(pivot["natural"] > 0, pivot["natural"], 0.0)
    pivot["amplify_score"] = pivot["amplify_benefit"] - args.lambda_collateral * pivot["C_icl_abs"]

    # Suppress heads that are natural promoters in intact model (ablation delta < 0).
    pivot["suppress_benefit"] = np.where(pivot["natural"] < 0, -pivot["natural"], 0.0)
    pivot["suppress_score"] = pivot["suppress_benefit"] - args.lambda_collateral * pivot["C_icl_abs"]

    pivot["recommended_action"] = "none"
    pivot.loc[pivot["amplify_score"] > pivot["suppress_score"], "recommended_action"] = "amplify"
    pivot.loc[pivot["suppress_score"] > pivot["amplify_score"], "recommended_action"] = "suppress"

    ranking_csv = out_dir / "selective_head_ranking.csv"
    pivot.sort_values(["amplify_score", "suppress_score"], ascending=False).to_csv(ranking_csv, index=False)

    amp = (
        pivot[pivot["amplify_benefit"] > 0]
        .sort_values("amplify_score", ascending=False)
        .head(args.top_k_amplify)
    )
    sup = (
        pivot[pivot["suppress_benefit"] > 0]
        .sort_values("suppress_score", ascending=False)
        .head(args.top_k_suppress)
    )

    policy = {
        "meta": {
            "input_csv": str(in_path),
            "latest_only": bool(args.latest_only),
            "lambda_collateral": args.lambda_collateral,
            "top_k_amplify": int(args.top_k_amplify),
            "top_k_suppress": int(args.top_k_suppress),
        },
        "interventions": [],
    }

    for row in amp.itertuples():
        policy["interventions"].append(
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "mode": "amplify",
                "factor": float(args.default_amplify_factor),
                "scores": {
                    "natural_delta": float(row.natural),
                    "icl_delta": float(row.icl),
                    "selectivity_score": float(row.amplify_score),
                },
            }
        )

    for row in sup.itertuples():
        policy["interventions"].append(
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "mode": "suppress",
                "factor": 0.0,
                "scores": {
                    "natural_delta": float(row.natural),
                    "icl_delta": float(row.icl),
                    "selectivity_score": float(row.suppress_score),
                },
            }
        )

    policy_json = out_dir / "selected_policy.json"
    policy_json.write_text(json.dumps(policy, indent=2))

    print(f"Saved ranking: {ranking_csv}")
    print(f"Saved policy: {policy_json}")
    print("Top amplify:")
    for row in amp.itertuples():
        print(f"  L{int(row.layer)}H{int(row.head)} score={row.amplify_score:.4f} nat={row.natural:.4f} icl={row.icl:.4f}")
    print("Top suppress:")
    for row in sup.itertuples():
        print(f"  L{int(row.layer)}H{int(row.head)} score={row.suppress_score:.4f} nat={row.natural:.4f} icl={row.icl:.4f}")


if __name__ == "__main__":
    main()
