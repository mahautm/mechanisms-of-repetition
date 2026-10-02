#!/usr/bin/env python3
"""Summarise the initialisation survey (run_init_survey_array.sh).

One row per (model, checkpoint, condition): greedy repetition rate on the paired,
length-matched prompt set, with Wilson 95% CIs, for the paper metric (repeats) and
with EOS-terminated generations not counted as cycles (repeats_no_eos). Seed families
(e.g. pythia-160m + pythia-160m-seed1..9) are also pooled: mean and range across seeds.
"""

import argparse
import glob
import os
import re

import pandas as pd

from analyze_per_prompt_mcnemar import wilson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="outputs_init_survey")
    ap.add_argument("--out_csv", default="outputs_init_survey/init_survey_summary.csv")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.input_dir, "*", "*", "*_per_prompt.csv.gz"))):
        model = os.path.basename(os.path.dirname(os.path.dirname(f)))
        d = pd.read_csv(f)
        d = d[d["head"] == -1]
        for (ck, cond), g in d.groupby(["checkpoint", "condition"]):
            n = len(g)
            r = {"model": model, "checkpoint": ck, "condition": cond, "n": n}
            for m in ["repeats", "repeats_no_eos"]:
                x = int(g[m].sum())
                lo, hi = wilson(x, n)
                r.update({m: x / n, f"{m}_lo": lo, f"{m}_hi": hi})
            r["eos_terminated"] = g["eos_terminated"].mean()
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    pd.set_option("display.width", 200)
    for _, r in df.sort_values(["checkpoint", "model", "condition"]).iterrows():
        print(f"{r.model:32s} {r.checkpoint:22s} {r.condition:7s} n={r.n:4d}  "
              f"rep={r.repeats:6.1%} [{r.repeats_lo:.1%},{r.repeats_hi:.1%}]  "
              f"rep_no_eos={r.repeats_no_eos:6.1%}  eos={r.eos_terminated:.1%}")

    # Pool seed families: strip "-seedN"; randinit seeds are grouped by model.
    df["family"] = df["model"].str.replace(r"-seed\d+$", "", regex=True)
    df["kind"] = df["checkpoint"].str.replace(r"randinit_seed\d+", "randinit", regex=True)
    fam = df.groupby(["family", "kind", "condition"]).agg(
        n_inits=("repeats", "size"), mean=("repeats", "mean"), min=("repeats", "min"), max=("repeats", "max"),
        mean_no_eos=("repeats_no_eos", "mean"), min_no_eos=("repeats_no_eos", "min"), max_no_eos=("repeats_no_eos", "max"),
    ).reset_index()
    print("\nAcross independent initialisations (seed families):")
    for _, r in fam.iterrows():
        print(f"  {r.family:30s} {r.kind:22s} {r.condition:7s} inits={r.n_inits:2d}  "
              f"rep mean={r['mean']:.1%} range [{r['min']:.1%}, {r['max']:.1%}]  "
              f"no_eos mean={r.mean_no_eos:.1%} range [{r.min_no_eos:.1%}, {r.max_no_eos:.1%}]")


if __name__ == "__main__":
    main()
