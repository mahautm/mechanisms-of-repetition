#!/usr/bin/env python3
"""
Analyze outputs from ablation_head_cycle_evolution.py.

Creates:
- per-checkpoint head rankings by ablation delta
- per-head aggregate summary across checkpoints
- cycle-size specific impact summary
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Analyze head-cycle ablation outputs")
    parser.add_argument("--overall_delta_csv", type=str, required=True)
    parser.add_argument("--cycle_delta_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_ablation_head_cycle")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    overall = pd.read_csv(args.overall_delta_csv)
    cycle = pd.read_csv(args.cycle_delta_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rankings = []
    for checkpoint, part in overall.groupby("checkpoint"):
        ranked = part.sort_values("delta_repetition_rate")
        top_down = ranked.head(args.top_k).copy()
        top_down["rank_type"] = "most_suppressive"
        top_up = ranked.tail(args.top_k).copy()
        top_up["rank_type"] = "most_inductive"
        top = pd.concat([top_down, top_up], ignore_index=True)
        checkpoint_rankings.append(top)

    checkpoint_rankings_df = pd.concat(checkpoint_rankings, ignore_index=True)

    head_global = (
        overall.groupby(["layer", "head"], as_index=False)
        .agg(
            mean_delta_repetition_rate=("delta_repetition_rate", "mean"),
            std_delta_repetition_rate=("delta_repetition_rate", "std"),
            mean_delta_cycle_count=("delta_avg_cycle_count", "mean"),
            n_checkpoints=("checkpoint", "nunique"),
        )
        .sort_values("mean_delta_repetition_rate")
    )

    cycle_summary = (
        cycle.groupby(["layer", "head", "cycle_size"], as_index=False)
        .agg(
            mean_delta_cycle_rate=("delta_rate_cycle_size_eq_k", "mean"),
            std_delta_cycle_rate=("delta_rate_cycle_size_eq_k", "std"),
            n_checkpoints=("checkpoint", "nunique"),
        )
        .sort_values(["cycle_size", "mean_delta_cycle_rate"])
    )

    checkpoint_rankings_path = output_dir / "ablation_checkpoint_rankings.csv"
    head_global_path = output_dir / "ablation_head_global_summary.csv"
    cycle_summary_path = output_dir / "ablation_cycle_size_summary.csv"

    checkpoint_rankings_df.to_csv(checkpoint_rankings_path, index=False)
    head_global.to_csv(head_global_path, index=False)
    cycle_summary.to_csv(cycle_summary_path, index=False)

    md_path = output_dir / "ablation_analysis_report.md"
    with open(md_path, "w") as f:
        f.write("# Ablation Head-Cycle Analysis Report\n\n")
        f.write("## Files\n\n")
        f.write(f"- Input overall deltas: {args.overall_delta_csv}\n")
        f.write(f"- Input cycle deltas: {args.cycle_delta_csv}\n")
        f.write(f"- Top-k per checkpoint: {checkpoint_rankings_path}\n")
        f.write(f"- Global head summary: {head_global_path}\n")
        f.write(f"- Cycle-size summary: {cycle_summary_path}\n\n")

        f.write("## Most Suppressive Heads (Global)\n\n")
        suppressive = head_global.head(min(10, len(head_global)))
        f.write(suppressive.to_string(index=False))
        f.write("\n\n")

        f.write("## Most Inductive Heads (Global)\n\n")
        inductive = head_global.tail(min(10, len(head_global))).sort_values(
            "mean_delta_repetition_rate", ascending=False
        )
        f.write(inductive.to_string(index=False))
        f.write("\n")

    print("Analysis complete.")
    print(f"Wrote: {checkpoint_rankings_path}")
    print(f"Wrote: {head_global_path}")
    print(f"Wrote: {cycle_summary_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
