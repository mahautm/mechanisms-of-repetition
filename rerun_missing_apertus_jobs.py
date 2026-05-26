#!/usr/bin/env python3
"""Find missing Apertus matrix outputs and print array indices to rerun."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_csv", type=str, default="outputs/head_policy_matrix/matrix.csv")
    parser.add_argument("--output_root", type=str, default="outputs/head_policy_eval/matrix")
    args = parser.parse_args()

    m = pd.read_csv(args.matrix_csv)
    missing = []
    for i, r in m.iterrows():
        if str(r["family"]).lower() != "apertus":
            continue
        safe_model = str(r["model_name"]).replace("/", "_")
        out = Path(args.output_root) / f"{r['family']}_{r['size']}" / str(r["hypothesis"]) / safe_model
        found = list(out.glob("*_noninferiority.csv"))
        if not found:
            missing.append(i)

    if not missing:
        print("No missing Apertus rows.")
        return

    print("Missing Apertus matrix indices:")
    print(",".join(str(x) for x in missing))


if __name__ == "__main__":
    main()
