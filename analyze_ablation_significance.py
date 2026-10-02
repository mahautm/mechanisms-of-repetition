#!/usr/bin/env python3
"""Significance testing for the per-head causal ablation and entropy-variability
results underlying Fig. 2 / Sec. 3.3 and Fig. 5 / Sec. 3.5.

Rerun this once the corrected-hook 1k-prompt sweep
(outputs_ablation_head_cycle_1k_pile/<model>/step*/chunk_*_of_*/*_icl_natural.csv)
has completed for all checkpoints. It aggregates counts across chunks, tests
each (layer, head, condition) for a shift in repetition rate under ablation,
and applies Bonferroni and Benjamini-Hochberg (FDR) correction across all
heads x conditions tested at a checkpoint. It also re-derives the Natural vs.
ICL entropy-variance comparison (Sec. 3.5) from entropy_evolution_results/.

No third-party dependencies (stdlib only), so it can run outside the `parr`
conda env -- e.g. directly on the login node while the GPU sweep is still
running on compute nodes.

Usage:
    python3 analyze_ablation_significance.py \\
        --ablation-dir outputs_ablation_head_cycle_1k_pile/pythia14b \\
        --entropy-json entropy_evolution_results/entropy_evolution_EleutherAI_pythia-1.4b.json \\
        --checkpoints step1 step1000 step36000 step71000 step143000

Output is plain text (safe to paste into the paper appendix) plus, with
--json-out, a machine-readable summary for downstream plotting.
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# Statistics (pure-Python; no scipy/numpy dependency)
# ---------------------------------------------------------------------------

def norm_sf(z: float) -> float:
    """Two-sided p-value for a standard normal z-statistic."""
    return math.erfc(abs(z) / math.sqrt(2))


def two_prop_ztest(x1: int, n1: int, x2: int, n2: int):
    """Independent-samples two-proportion z-test.

    Used instead of a paired test (e.g. McNemar's) because the ablation
    pipeline currently stores per-chunk aggregate repetition rates rather
    than per-prompt baseline/ablated outcome pairs. This is the more
    conservative choice: switch to McNemar's test once per-prompt pairing
    is logged (see NOTE at the bottom of this file).
    """
    if n1 == 0 or n2 == 0:
        return None, None
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return 0.0, 1.0
    z = (p2 - p1) / denom
    return z, norm_sf(z)


def bh_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg: returns the set of indices into `pvals` that pass FDR control."""
    m = len(pvals)
    if m == 0:
        return set()
    order = sorted(range(m), key=lambda i: pvals[i])
    largest_rank_passed = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank / m) * alpha:
            largest_rank_passed = rank
    return {order[i] for i in range(largest_rank_passed)}


def _betacf(a, b, x, maxit=200, eps=1e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    bt = math.exp(lbeta)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def f_test_two_sided(var1, n1, var2, n2):
    """Two-sided F-test on a ratio of variances (normal-theory; use as a sanity
    check alongside Brown-Forsythe, which is robust to non-normal entropy
    distributions but needs raw per-prompt values -- see run_brown_forsythe below)."""
    d1, d2 = n1 - 1, n2 - 1
    F = var1 / var2
    x = d2 / (d2 + d1 * F)
    cdf = _betai(d2 / 2.0, d1 / 2.0, x)
    return F, 2 * min(cdf, 1 - cdf)


def brown_forsythe(sample1, sample2):
    """Levene's test with median centering (Brown-Forsythe), robust to non-normality.
    Requires raw per-prompt values; falls back is handled by the caller."""
    def abs_dev_from_median(xs):
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        med = xs_sorted[n // 2] if n % 2 else 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2])
        return [abs(x - med) for x in xs]

    z1, z2 = abs_dev_from_median(sample1), abs_dev_from_median(sample2)
    n1, n2 = len(z1), len(z2)
    N = n1 + n2
    mean1, mean2 = sum(z1) / n1, sum(z2) / n2
    grand_mean = (sum(z1) + sum(z2)) / N
    ss_between = n1 * (mean1 - grand_mean) ** 2 + n2 * (mean2 - grand_mean) ** 2
    ss_within = sum((z - mean1) ** 2 for z in z1) + sum((z - mean2) ** 2 for z in z2)
    df1, df2 = 1, N - 2
    if ss_within == 0:
        return None, None
    F = (ss_between / df1) / (ss_within / df2)
    x = df2 / (df2 + df1 * F)
    p = 1 - _betai(df2 / 2.0, df1 / 2.0, x)
    return F, p


# ---------------------------------------------------------------------------
# Ablation significance
# ---------------------------------------------------------------------------

def load_checkpoint_csvs(step_dir):
    data = defaultdict(list)
    for chunk_dir in sorted(glob.glob(os.path.join(step_dir, "chunk_*_of_*"))):
        for fpath in glob.glob(os.path.join(chunk_dir, "*_icl_natural.csv")):
            with open(fpath, newline="") as fh:
                for row in csv.DictReader(fh):
                    key = (int(row["layer"]), int(row["head"]), row["condition"])
                    data[key].append((
                        int(row["n_prompts"]),
                        float(row["baseline_repetition_rate"]),
                        float(row["ablated_repetition_rate"]),
                    ))
    return data


def aggregate_counts(data):
    out = {}
    for key, rows in data.items():
        x_base = sum(round(n * br) for n, br, ar in rows)
        x_abl = sum(round(n * ar) for n, br, ar in rows)
        N = sum(n for n, br, ar in rows)
        out[key] = (x_base, x_abl, N)
    return out


def analyze_checkpoint(step_dir, alpha=0.05):
    data = load_checkpoint_csvs(step_dir)
    if not data:
        return None
    agg = aggregate_counts(data)
    heads = sorted({(l, h) for (l, h, _c) in agg})

    per_head = []
    for (l, h) in heads:
        entry = {"layer": l, "head": h}
        for cond in ("natural", "icl"):
            key = (l, h, cond)
            if key not in agg:
                continue
            x_base, x_abl, N = agg[key]
            z, p = two_prop_ztest(x_base, N, x_abl, N)
            entry[cond] = {
                "x_base": x_base, "x_abl": x_abl, "N": N,
                "delta": (x_abl - x_base) / N if N else None,
                "z": z, "p": p,
            }
        per_head.append(entry)

    all_p, all_labels = [], []
    S_vals = []
    for entry in per_head:
        if "natural" in entry and "icl" in entry:
            S_vals.append(0.5 * (entry["icl"]["delta"] + entry["natural"]["delta"]))
        for cond in ("natural", "icl"):
            if cond in entry:
                all_p.append(entry[cond]["p"])
                all_labels.append((entry["layer"], entry["head"], cond))

    n_tests = len(all_p)
    n_uncorrected = sum(1 for p in all_p if p < alpha)
    bonf_alpha = alpha / n_tests if n_tests else None
    n_bonf = sum(1 for p in all_p if p < bonf_alpha) if n_tests else 0
    bh_idx = bh_correction(all_p, alpha=alpha)
    bh_heads = sorted({all_labels[i] for i in bh_idx})

    return {
        "n_heads": len(heads),
        "n_tests": n_tests,
        "n_uncorrected": n_uncorrected,
        "bonferroni_alpha": bonf_alpha,
        "n_bonferroni": n_bonf,
        "n_bh": len(bh_idx),
        "bh_significant_heads": bh_heads,
        "max_abs_S": max((abs(s) for s in S_vals), default=None),
        "mean_abs_S": (sum(abs(s) for s in S_vals) / len(S_vals)) if S_vals else None,
    }


# ---------------------------------------------------------------------------
# Entropy variability (Sec. 3.5 / Fig. 5)
# ---------------------------------------------------------------------------

def analyze_entropy(entropy_json_path, checkpoints):
    with open(entropy_json_path) as f:
        data = json.load(f)

    results = {}
    for step in checkpoints:
        if step not in data:
            continue
        sl = data[step]
        # Prefer raw per-prompt values if the logging has been extended to keep them.
        raw_natural = sl.get("natural_raw") or sl.get("natural_entropies_raw")
        raw_icl = sl.get("no_cycle_raw") or sl.get("no_cycle_entropies_raw")
        if raw_natural and raw_icl:
            F_bf, p_bf = brown_forsythe(raw_natural, raw_icl)
            results[step] = {"method": "brown_forsythe", "F": F_bf, "p": p_bf,
                              "n_natural": len(raw_natural), "n_icl": len(raw_icl)}
            continue

        n_std, n_n = sl.get("natural_std"), sl.get("natural_count")
        i_std, i_n = sl.get("no_cycle_std"), sl.get("no_cycle_count")
        if None in (n_std, n_n, i_std, i_n) or n_n < 2 or i_n < 2:
            continue
        F, p = f_test_two_sided(n_std ** 2, n_n, i_std ** 2, i_n)
        results[step] = {"method": "f_test_on_reported_std", "F": F, "p": p,
                          "n_natural": n_n, "n_icl": i_n,
                          "natural_std": n_std, "icl_std": i_std}
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", required=True,
                     help="Directory containing step*/chunk_*_of_*/*_icl_natural.csv (per model).")
    ap.add_argument("--checkpoints", nargs="+",
                     default=["step1", "step1000", "step36000", "step71000", "step143000"])
    ap.add_argument("--entropy-json", default=None,
                     help="Path to entropy_evolution_results/entropy_evolution_<model>.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json-out", default=None, help="Optional path to dump machine-readable results.")
    args = ap.parse_args()

    summary = {"ablation": {}, "entropy": {}}

    print("=" * 70)
    print("Per-head ablation significance (Fig. 2 / Sec. 3.3)")
    print("=" * 70)
    for step in args.checkpoints:
        step_dir = os.path.join(args.ablation_dir, step)
        if not os.path.isdir(step_dir):
            print(f"{step}: directory not found, skipping ({step_dir})")
            continue
        res = analyze_checkpoint(step_dir, alpha=args.alpha)
        if res is None:
            print(f"{step}: no CSVs found under {step_dir} (sweep not yet complete?)")
            continue
        summary["ablation"][step] = res
        print(f"\n{step}: {res['n_heads']} heads, {res['n_tests']} head x condition tests")
        print(f"  uncorrected p<{args.alpha}: {res['n_uncorrected']}/{res['n_tests']} "
              f"({100 * res['n_uncorrected'] / res['n_tests']:.1f}%)")
        print(f"  Bonferroni (alpha={res['bonferroni_alpha']:.2e}): {res['n_bonferroni']} significant")
        print(f"  Benjamini-Hochberg (q={args.alpha}): {res['n_bh']} significant")
        if res["n_bh"]:
            heads_str = ", ".join(f"L{l}.H{h}({c})" for l, h, c in res["bh_significant_heads"][:20])
            print(f"    -> {heads_str}{' ...' if res['n_bh'] > 20 else ''}")
        print(f"  max|S| = {res['max_abs_S']:.4f}, mean|S| = {res['mean_abs_S']:.4f}")

    if args.entropy_json:
        print("\n" + "=" * 70)
        print("Entropy variability, Natural vs. ICL (Fig. 5 / Sec. 3.5)")
        print("=" * 70)
        ent = analyze_entropy(args.entropy_json, args.checkpoints)
        summary["entropy"] = ent
        for step, r in ent.items():
            print(f"\n{step} [{r['method']}]: F={r['F']:.3f}, p={r['p']:.4g} "
                  f"(n_natural={r['n_natural']}, n_icl={r['n_icl']})")
            if r["method"] == "f_test_on_reported_std":
                print("  NOTE: computed from stored mean/std, not raw per-prompt values -- "
                      "assumes approximate normality. Add raw-value logging to entropy "
                      "collection and rerun for the more robust Brown-Forsythe test.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote machine-readable summary to {args.json_out}")


if __name__ == "__main__":
    main()

# NOTE on upgrading to a paired test:
# `ablation_head_cycle_icl_natural.py` currently discards per-prompt outcomes
# after computing the aggregate repetition rate (see `compute_repetition_metrics`
# call sites). To enable McNemar's test (more powerful than the independent
# two-proportion test used here, since baseline and ablated generations share
# the same prompt set), extend that script to also save a per-prompt boolean
# "is_cyclical" column for both baseline and ablated runs, then swap
# `two_prop_ztest` for a McNemar implementation keyed on the four
# concordant/discordant counts.
