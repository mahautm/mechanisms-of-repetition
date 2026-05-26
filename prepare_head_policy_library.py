#!/usr/bin/env python3
"""Prepare policy library across hypotheses and transfer groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_policy(df: pd.DataFrame, lam: float, top_k_amp: int, top_k_sup: int, amp_factor: float) -> dict:
    grouped = (
        df.groupby(["layer", "head", "condition"], as_index=False)
        .agg(delta_rep=("delta_repetition_rate", "mean"))
    )
    pivot = grouped.pivot_table(index=["layer", "head"], columns="condition", values="delta_rep", aggfunc="mean").reset_index()
    if "natural" not in pivot.columns or "icl" not in pivot.columns:
        raise ValueError("Both natural and icl conditions are required")

    pivot["abs_icl"] = pivot["icl"].abs()
    pivot["amp_benefit"] = pivot["natural"].clip(lower=0)
    pivot["amp_score"] = pivot["amp_benefit"] - lam * pivot["abs_icl"]
    pivot["sup_benefit"] = (-pivot["natural"]).clip(lower=0)
    pivot["sup_score"] = pivot["sup_benefit"] - lam * pivot["abs_icl"]

    amp = pivot[pivot["amp_benefit"] > 0].sort_values("amp_score", ascending=False).head(top_k_amp)
    sup = pivot[pivot["sup_benefit"] > 0].sort_values("sup_score", ascending=False).head(top_k_sup)

    interventions = []
    for r in amp.itertuples():
        interventions.append({"layer": int(r.layer), "head": int(r.head), "mode": "amplify", "factor": float(amp_factor)})
    for r in sup.itertuples():
        interventions.append({"layer": int(r.layer), "head": int(r.head), "mode": "suppress", "factor": 0.0})

    return {"interventions": interventions}


def write_policy_suite(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("policy_lambda1.0_k2_factor1.25.json", 1.0, 2, 2, 1.25),
        ("policy_lambda2.0_k2_factor1.10.json", 2.0, 2, 2, 1.10),
        ("policy_lambda0.5_k4_factor1.50.json", 0.5, 4, 4, 1.50),
    ]
    for name, lam, ka, ks, fac in configs:
        p = build_policy(df, lam=lam, top_k_amp=ka, top_k_sup=ks, amp_factor=fac)
        (out_dir / name).write_text(json.dumps(p, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs/head_policy_library")
    parser.add_argument("--pythia70_csv", type=str, default="plots/ablation_pythia70m/combined_ablation_effects.csv")
    parser.add_argument("--pythia14_csv", type=str, default="plots/ablation_pythia14b/combined_ablation_effects.csv")
    parser.add_argument("--olmo1b_csv", type=str, default="plots/ablation_olmo1b/combined_ablation_effects.csv")
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    p70 = pd.read_csv(args.pythia70_csv)
    p14 = pd.read_csv(args.pythia14_csv)
    o1b = pd.read_csv(args.olmo1b_csv)

    write_policy_suite(p70, root / "pythia70m")
    write_policy_suite(p14, root / "pythia14b")
    write_policy_suite(p14, root / "pythia14b_transfer")
    write_policy_suite(o1b, root / "olmo1b")
    write_policy_suite(o1b, root / "olmo1b_transfer")
    write_policy_suite(o1b, root / "apertus_transfer")

    print(f"Saved policy library under: {root}")


if __name__ == "__main__":
    main()
