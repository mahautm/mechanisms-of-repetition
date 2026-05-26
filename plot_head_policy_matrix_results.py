#!/usr/bin/env python3
"""Aggregate matrix outputs and generate summary plots/tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="outputs/head_policy_eval/matrix")
    parser.add_argument("--output_root", type=str, default="plots/head_policy_matrix")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = list(input_root.glob("**/*_noninferiority.csv"))
    if not files:
        raise SystemExit(f"No noninferiority csv files under {input_root}")

    dfs = []
    for f in files:
        d = pd.read_csv(f)
        parts = f.relative_to(input_root).parts
        # matrix/family_size/hypothesis/model/file.csv
        if len(parts) >= 4:
            d["family_size"] = parts[0]
            d["hypothesis"] = parts[1]
            d["model_tag"] = parts[2]
        d["source_file"] = str(f)
        dfs.append(d)

    all_df = pd.concat(dfs, ignore_index=True)
    all_csv = output_root / "matrix_noninferiority_all.csv"
    all_df.to_csv(all_csv, index=False)

    pass_rate = (
        all_df.groupby(["arm"], as_index=False)
        .agg(pass_rate=("pass_noninferiority", "mean"), n=("pass_noninferiority", "size"))
        .sort_values("pass_rate", ascending=False)
    )
    pass_csv = output_root / "arm_pass_rates.csv"
    pass_rate.to_csv(pass_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(pass_rate["arm"], pass_rate["pass_rate"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Pass non-inferiority rate")
    plt.title("Non-Inferiority Pass Rate by Arm")
    plt.tight_layout()
    bar_png = output_root / "arm_pass_rates.png"
    plt.savefig(bar_png, dpi=180)
    plt.close()

    pareto = all_df[["arm", "nat_reduction_rel", "icl_change_abs"]].copy()
    pareto = pareto.dropna()

    plt.figure(figsize=(7, 6))
    for arm, sub in pareto.groupby("arm"):
        plt.scatter(sub["icl_change_abs"], sub["nat_reduction_rel"], label=arm, alpha=0.7)
    plt.xlabel("ICL harm |Δ| (lower better)")
    plt.ylabel("Natural reduction (higher better)")
    plt.title("Pareto-style Tradeoff Across Matrix Runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    pareto_png = output_root / "pareto_scatter.png"
    plt.savefig(pareto_png, dpi=180)
    plt.close()

    report = output_root / "REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# Head Policy Matrix Report",
                "",
                f"- Inputs: {input_root}",
                f"- Aggregated rows: {len(all_df)}",
                f"- Non-inferiority table: {all_csv}",
                f"- Arm pass rates: {pass_csv}",
                f"- Plots: {bar_png}, {pareto_png}",
                "",
                "## Pass Rates by Arm",
                "",
                df_to_markdown(pass_rate),
            ]
        )
    )

    print(f"Saved: {all_csv}")
    print(f"Saved: {pass_csv}")
    print(f"Saved: {bar_png}")
    print(f"Saved: {pareto_png}")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
