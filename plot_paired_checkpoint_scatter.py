#!/usr/bin/env python3
"""Paper-style checkpoint scatter summary (combine_checkpoint_scatters.py) for the paired fp16 sweep.

Converts the per-head table from analyze_per_prompt_mcnemar.py into the combined_ablation_effects.csv
layout and reuses build_montage, so the figure matches the paper version panel for panel.
"""

import argparse
from pathlib import Path

import pandas as pd

from combine_checkpoint_scatters import build_montage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per_head_csv", default="outputs_ablation_paired_fp16/pythia14b/per_head_mcnemar_all.csv")
    ap.add_argument("--out_dir", default="plots/ablation_pythia14b_paired")
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        default=["step1", "step1000", "step36000", "step71000", "step143000"],
        help="Four small panels plus the final (large) one; the paper figure omits step0.",
    )
    ap.add_argument("--axis_limit", type=float, default=None, help="Fixed ±limit so figures are comparable")
    ap.add_argument("--model_name", default=None, help="Legend label")
    args = ap.parse_args()

    df = pd.read_csv(args.per_head_csv)
    df = df[df["checkpoint"].isin(args.checkpoints)]
    out = pd.DataFrame({
        "checkpoint": df["checkpoint"],
        "head": df["head"],
        "condition": df["condition"],
        "delta_repetition_rate": df["delta"],
        "layer": df["layer"],
        "source": "paired_fp16",
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "combined_ablation_effects.csv", index=False)
    print(f"Saved {build_montage(out_dir, 'rep_rate', args.axis_limit, args.model_name)}")


if __name__ == "__main__":
    main()
