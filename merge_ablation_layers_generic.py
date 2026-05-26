#!/usr/bin/env python3
"""Merge per-layer ablation CSV files for a given model into one raw CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge layer-wise ablation outputs")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--expected_layers", type=int, required=True)
    parser.add_argument("--expected_checkpoints", type=str, default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    safe_model = safe_model_name(args.model_name)
    files = sorted(input_dir.rglob(f"head_cycle_ablation_{safe_model}_L*_icl_natural.csv"))
    if not files:
        raise FileNotFoundError(f"No layer CSVs found in {input_dir} for model {args.model_name}")

    frames = [pd.read_csv(f) for f in files]
    merged = pd.concat(frames, ignore_index=True)

    layers_present = sorted(int(x) for x in merged["layer"].dropna().unique().tolist())
    missing_layers = sorted(set(range(args.expected_layers)).difference(layers_present))
    if missing_layers:
        raise SystemExit(f"Missing layers: {missing_layers}")

    normalized_ckpts = args.expected_checkpoints.replace("|", ",")
    expected_checkpoints = [x.strip() for x in normalized_ckpts.split(",") if x.strip()]
    if expected_checkpoints:
        got = set(merged["checkpoint"].dropna().astype(str).unique().tolist())
        missing_ckpts = [ck for ck in expected_checkpoints if ck not in got]
        if missing_ckpts:
            raise SystemExit(f"Missing checkpoints: {missing_ckpts}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"Merged files: {len(files)}")
    print(f"Rows: {len(merged)}")
    print(f"Layers present ({len(layers_present)}): {layers_present[0]}..{layers_present[-1]}")
    print(f"Saved merged CSV: {output_csv}")


if __name__ == "__main__":
    main()
