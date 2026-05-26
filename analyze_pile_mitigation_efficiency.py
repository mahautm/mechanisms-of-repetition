#!/usr/bin/env python3
"""Analyze mitigation effectiveness vs efficiency on Pile reruns.

Expected input folders under outputs/mitigations_pile:
- <model>_greedy_v2
- <model>_p0.5_v2
- <model>_p0.9_v2

Each folder should contain pile_eval_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_summary(path: Path) -> Dict:
    return json.loads(path.read_text())


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a simple markdown table without requiring tabulate."""
    headers = list(df.columns)
    rows = [headers]
    rows.extend(df.astype(str).itertuples(index=False, name=None))

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    def fmt_row(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"

    header_line = fmt_row(headers)
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    body_lines = [fmt_row(row) for row in rows[1:]]
    return "\n".join([header_line, sep_line, *body_lines])


def main() -> None:
    base = Path("/home/mmahaut/projects/parrots/outputs/mitigations_pile")
    rows: List[Dict] = []

    model_keys = sorted(
        set(
            p.name.replace("_greedy_v2", "")
            for p in base.glob("*_greedy_v2")
            if (p / "pile_eval_summary.json").exists()
        )
    )

    for model in model_keys:
        baseline_path = base / f"{model}_greedy_v2" / "pile_eval_summary.json"
        p05_path = base / f"{model}_p0.5_v2" / "pile_eval_summary.json"
        p09_path = base / f"{model}_p0.9_v2" / "pile_eval_summary.json"

        if not baseline_path.exists():
            continue

        baseline = load_summary(baseline_path)
        comparisons = []
        if p05_path.exists():
            comparisons.append(("p0.5", load_summary(p05_path)))
        if p09_path.exists():
            comparisons.append(("p0.9", load_summary(p09_path)))

        b_nat = baseline["natural"]
        b_icl = baseline["icl"]

        for cond, cur in comparisons:
            c_nat = cur["natural"]
            c_icl = cur["icl"]

            # Positive means mitigation reduced cyclical behavior vs baseline.
            delta_nat_cycle = b_nat["cyclical_rate"] - c_nat["cyclical_rate"]
            delta_icl_cycle = b_icl["cyclical_rate"] - c_icl["cyclical_rate"]
            delta_nat_cycle_length = b_nat["mean_cycle_length"] - c_nat["mean_cycle_length"]
            delta_icl_cycle_length = b_icl["mean_cycle_length"] - c_icl["mean_cycle_length"]

            # Throughput ratio > 1 means condition is faster than baseline.
            nat_tps_ratio = (
                c_nat["tokens_per_second"] / b_nat["tokens_per_second"]
                if b_nat["tokens_per_second"] > 0
                else 0.0
            )
            icl_tps_ratio = (
                c_icl["tokens_per_second"] / b_icl["tokens_per_second"]
                if b_icl["tokens_per_second"] > 0
                else 0.0
            )

            # Simple efficiency: cycle reduction per second of runtime.
            nat_efficiency = (
                delta_nat_cycle / c_nat["elapsed_seconds"] if c_nat["elapsed_seconds"] > 0 else 0.0
            )
            icl_efficiency = (
                delta_icl_cycle / c_icl["elapsed_seconds"] if c_icl["elapsed_seconds"] > 0 else 0.0
            )

            rows.append(
                {
                    "model": model,
                    "condition": cond,
                    "delta_nat_cyclical_rate_vs_greedy": delta_nat_cycle,
                    "delta_icl_cyclical_rate_vs_greedy": delta_icl_cycle,
                    "delta_nat_mean_cycle_length_vs_greedy": delta_nat_cycle_length,
                    "delta_icl_mean_cycle_length_vs_greedy": delta_icl_cycle_length,
                    "nat_tokens_per_second_ratio_vs_greedy": nat_tps_ratio,
                    "icl_tokens_per_second_ratio_vs_greedy": icl_tps_ratio,
                    "nat_efficiency_cycle_reduction_per_sec": nat_efficiency,
                    "icl_efficiency_cycle_reduction_per_sec": icl_efficiency,
                }
            )

    if not rows:
        print("No completed v2 summaries found yet.")
        return

    df = pd.DataFrame(rows)
    out_csv = Path("/home/mmahaut/projects/parrots/outputs/mitigations_pile/mitigation_efficiency_v2.csv")
    out_md = Path("/home/mmahaut/projects/parrots/outputs/mitigations_pile/mitigation_efficiency_v2.md")

    df.to_csv(out_csv, index=False)
    out_md.write_text(df_to_markdown(df))

    print(f"Saved {out_csv}")
    print(f"Saved {out_md}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()