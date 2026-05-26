#!/usr/bin/env python3
"""Merge per-layer Apertus ablation CSVs into a single combined raw CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge layer-wise Apertus ablation outputs")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/mmahaut/projects/parrots/outputs_ablation_head_cycle_apertus_all_layers",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/mmahaut/projects/parrots/outputs_ablation_head_cycle_apertus_all_layers/head_cycle_ablation_swiss-ai_Apertus-8B-2509_ALL_icl_natural.csv",
    )
    parser.add_argument("--expected_layers", type=int, default=32)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    files = sorted(input_dir.glob("head_cycle_ablation_swiss-ai_Apertus-8B-2509_L*_icl_natural.csv"))
    if not files:
        raise FileNotFoundError(f"No layer CSVs found in {input_dir}")

    frames = [pd.read_csv(f) for f in files]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(args.output_csv, index=False)

    layers_present = sorted(int(x) for x in merged["layer"].dropna().unique().tolist())
    print(f"Merged files: {len(files)}")
    print(f"Rows: {len(merged)}")
    print(f"Layers present ({len(layers_present)}): {layers_present[:5]} ... {layers_present[-5:]}")

    if len(layers_present) != args.expected_layers:
        missing = sorted(set(range(args.expected_layers)).difference(layers_present))
        raise SystemExit(f"Missing layers: {missing}")

    print(f"Saved merged CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
