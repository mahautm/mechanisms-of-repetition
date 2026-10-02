#!/usr/bin/env python3
"""Paired (per-prompt) statistics for the head-ablation sweep.

Consumes the *_per_prompt.csv.gz files written by ablation_head_cycle_icl_natural.py
(head = -1: unablated baseline, head = -2: baseline replicate, head >= 0: ablated head).

Per checkpoint it reports:
  1. Baseline repetition rate per condition, with Wilson 95% CIs.
  2. Noise floor: discordance between baseline and its replicate (GPU nondeterminism).
  3. Per-head ablation effects: delta = P(repeat | ablated) - P(repeat | baseline) on the same
     prompts, paired 95% CI, exact McNemar test, Bonferroni and Benjamini-Hochberg correction
     across all heads x conditions at that checkpoint. delta > 0 means the head SUPPRESSES
     repetition (removing it increases repetition); delta < 0 means it FACILITATES repetition.
  4. Condition-level mean delta over all heads, CI over prompts (heads share prompts, so
     prompts are the independent unit).
  5. (--paired only) Natural vs. ICL paired contrasts on the same source texts:
     baseline repetition (McNemar), mean ablation effect (Wilcoxon signed-rank), and per head.

Usage:
    python analyze_per_prompt_mcnemar.py --input_dir outputs_ablation_paired/pythia14b \\
        --checkpoints step0 step1 step1000 step36000 step71000 step143000 --paired \\
        --out_csv outputs_ablation_paired/pythia14b/per_head_mcnemar.csv
"""

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy import stats


def wilson(x, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    return stats.binomtest(min(b, c), b + c, 0.5).pvalue


def bh(pvals, alpha):
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, m + 1) / m
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, dtype=bool)
    sig[order[:k]] = True
    return sig


def load(input_dir, checkpoint):
    safe = checkpoint.replace("/", "_").replace(":", "_")
    files = glob.glob(os.path.join(input_dir, safe, "**", "*_per_prompt.csv.gz"), recursive=True)
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired_delta_ci(base, abl):
    """Mean paired difference abl - base (0/1 outcomes), normal-approx 95% CI."""
    d = abl.astype(float) - base.astype(float)
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return m, m - 1.96 * se, m + 1.96 * se


def analyze_checkpoint(df, checkpoint, metric, alpha, paired):
    df = df.copy()
    df["chunk_index"] = df["chunk_index"].fillna(-1).astype(int)
    key = ["condition", "layer", "chunk_index", "prompt_idx"]

    base = df[df["head"] == -1][key + [metric]].rename(columns={metric: "base"})
    rep = df[df["head"] == -2][key + [metric]].rename(columns={metric: "rep"})
    abl = df[df["head"] >= 0][key + ["head", metric]].rename(columns={metric: "abl"})
    abl = abl.merge(base, on=key, how="inner")

    print(f"\n=== {checkpoint}  (metric: {metric}) ===")

    # 1. Baseline rates: one baseline per (condition, chunk, prompt); use the lowest layer run.
    first_layer = base.groupby(["condition", "chunk_index"])["layer"].transform("min")
    b1 = base[base["layer"] == first_layer]
    for cond, g in b1.groupby("condition"):
        n, x = len(g), int(g["base"].sum())
        lo, hi = wilson(x, n)
        print(f"  baseline {cond:7s}: {x}/{n} = {x / n:.1%}  [95% CI {lo:.1%}, {hi:.1%}]")

    # 2. Noise floor from the replicate baseline.
    if len(rep):
        nr = base.merge(rep, on=key)
        for cond, g in nr.groupby("condition"):
            b = int(((g["base"] == 1) & (g["rep"] == 0)).sum())
            c = int(((g["base"] == 0) & (g["rep"] == 1)).sum())
            print(f"  noise floor {cond:7s}: discordant {(b + c) / len(g):.2%} of {len(g)} prompt-runs "
                  f"(b={b}, c={c}, McNemar p={mcnemar_exact(b, c):.3g})")

    if abl.empty:
        return pd.DataFrame()

    # 3. Per-head McNemar.
    rows = []
    for (cond, layer, head), g in abl.groupby(["condition", "layer", "head"]):
        b = int(((g["base"] == 1) & (g["abl"] == 0)).sum())
        c = int(((g["base"] == 0) & (g["abl"] == 1)).sum())
        m, lo, hi = paired_delta_ci(g["base"].values, g["abl"].values)
        rows.append({
            "checkpoint": checkpoint, "condition": cond, "layer": layer, "head": head, "n": len(g),
            "base_rate": g["base"].mean(), "abl_rate": g["abl"].mean(),
            "b_rep_to_norep": b, "c_norep_to_rep": c,
            "delta": m, "ci_lo": lo, "ci_hi": hi, "p_mcnemar": mcnemar_exact(b, c),
        })
    res = pd.DataFrame(rows)
    res["sig_bonf"] = res["p_mcnemar"] < alpha / len(res)
    res["sig_bh"] = bh(res["p_mcnemar"].values, alpha)
    res["direction"] = np.where(res["delta"] > 0, "suppressor", np.where(res["delta"] < 0, "facilitator", "none"))

    print(f"  per-head tests: {len(res)} (heads x conditions); Bonferroni alpha={alpha / len(res):.2e}")
    print("  counts are suppressors (ablation raises repetition) / facilitators (ablation lowers it)")
    for cond, g in res.groupby("condition"):
        cnt = lambda mask: f"{int((mask & (g.delta > 0)).sum())} / {int((mask & (g.delta < 0)).sum())}"
        print(f"  {cond:7s}: nominal p<{alpha}: {cnt(g.p_mcnemar < alpha)} | BH: {cnt(g.sig_bh)} | Bonferroni: {cnt(g.sig_bonf)}")
        top = g.reindex(g["delta"].abs().sort_values(ascending=False).index).head(5)
        for _, r in top.iterrows():
            print(f"      L{int(r.layer)}H{int(r['head'])}: delta={r.delta * 100:+.1f}pp "
                  f"[{r.ci_lo * 100:+.1f}, {r.ci_hi * 100:+.1f}]  b={r.b_rep_to_norep} c={r.c_norep_to_rep} "
                  f"p={r.p_mcnemar:.2g} {r.direction} {'BH*' if r.sig_bh else ''}")

    # 4. Condition-level mean effect; prompts are the independent unit.
    abl["d"] = abl["abl"] - abl["base"]
    per_prompt = abl.groupby(["condition", "chunk_index", "prompt_idx"])["d"].mean().reset_index()
    for cond, g in per_prompt.groupby("condition"):
        m = g["d"].mean()
        se = g["d"].std(ddof=1) / math.sqrt(len(g))
        print(f"  mean delta over all heads, {cond:7s}: {m * 100:+.2f}pp "
              f"[95% CI {(m - 1.96 * se) * 100:+.2f}, {(m + 1.96 * se) * 100:+.2f}] (n prompts={len(g)})")

    # 5. Paired Natural vs. ICL contrasts on the same source texts.
    if paired:
        bb = b1.pivot_table(index=["chunk_index", "prompt_idx"], columns="condition", values="base").dropna()
        if {"natural", "icl"} <= set(bb.columns):
            b = int(((bb.natural == 1) & (bb.icl == 0)).sum())
            c = int(((bb.natural == 0) & (bb.icl == 1)).sum())
            m, lo, hi = paired_delta_ci(bb.natural.values, bb.icl.values)
            print(f"  paired baseline ICL - Natural: {m * 100:+.1f}pp [{lo * 100:+.1f}, {hi * 100:+.1f}], "
                  f"McNemar p={mcnemar_exact(b, c):.3g} (n sources={len(bb)})")
        pp = per_prompt.pivot_table(index=["chunk_index", "prompt_idx"], columns="condition", values="d").dropna()
        if {"natural", "icl"} <= set(pp.columns):
            diff = pp.icl - pp.natural
            m = diff.mean()
            se = diff.std(ddof=1) / math.sqrt(len(diff))
            p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            print(f"  paired mean-ablation-effect ICL - Natural: {m * 100:+.2f}pp "
                  f"[{(m - 1.96 * se) * 100:+.2f}, {(m + 1.96 * se) * 100:+.2f}], Wilcoxon p={p:.3g}")
        # Per head: does ablating this head affect ICL differently from Natural on the same sources?
        ph = abl.pivot_table(index=["layer", "head", "chunk_index", "prompt_idx"], columns="condition", values="d").dropna()
        if {"natural", "icl"} <= set(ph.columns):
            ph = (ph.icl - ph.natural).rename("x").reset_index()
            hrows = []
            for (layer, head), g in ph.groupby(["layer", "head"]):
                x = g["x"].values
                p = stats.wilcoxon(x).pvalue if (x != 0).any() else 1.0
                hrows.append((layer, head, x.mean(), p))
            hp = pd.DataFrame(hrows, columns=["layer", "head", "icl_minus_nat", "p_icl_vs_nat"])
            hp["sig_bh_icl_vs_nat"] = bh(hp["p_icl_vs_nat"].values, alpha)
            print(f"  per-head ICL vs Natural contrast: {int(hp.sig_bh_icl_vs_nat.sum())}/{len(hp)} heads BH-significant")
            for _, r in hp[hp.sig_bh_icl_vs_nat].sort_values("p_icl_vs_nat").head(10).iterrows():
                print(f"      L{int(r.layer)}H{int(r['head'])}: ICL-Nat delta diff={r.icl_minus_nat * 100:+.1f}pp p={r.p_icl_vs_nat:.2g}")
            res = res.merge(hp, on=["layer", "head"], how="left")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--metric", default="repeats", choices=["repeats", "repeats_no_eos"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--paired", action="store_true", help="Inputs come from --prompt_mode paired")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    all_res = []
    for ck in args.checkpoints:
        df = load(args.input_dir, ck)
        if df is None:
            print(f"\n=== {ck}: no per-prompt files, skipped ===")
            continue
        all_res.append(analyze_checkpoint(df, ck, args.metric, args.alpha, args.paired))
    if args.out_csv and all_res:
        pd.concat(all_res, ignore_index=True).to_csv(args.out_csv, index=False)
        print(f"\nSaved per-head table: {args.out_csv}")


if __name__ == "__main__":
    main()
