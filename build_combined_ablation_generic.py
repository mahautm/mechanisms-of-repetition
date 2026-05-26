#!/usr/bin/env python3
"""Build combined_ablation_effects.csv from merged raw ablation output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    expected = [
        "checkpoint",
        "head",
        "condition",
        "delta_repetition_rate",
        "delta_avg_cycle_count",
        "layer",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}")

    combined = df[expected].copy()
    combined["source"] = "icl_natural"
    combined_path = out_dir / "combined_ablation_effects.csv"
    combined.to_csv(combined_path, index=False)

    summary_lines = [
        "# Condition-Aware Ablation Summary",
        "",
        f"- Total rows: {len(combined)}",
        f"- Conditions present: {', '.join(sorted(combined['condition'].dropna().unique().tolist()))}",
        f"- Checkpoints present: {', '.join(sorted(combined['checkpoint'].dropna().astype(str).unique().tolist()))}",
    ]
    summary_path = out_dir / "ablation_condition_summary.md"
    summary_path.write_text("\n".join(summary_lines))

    print(f"Saved: {combined_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
