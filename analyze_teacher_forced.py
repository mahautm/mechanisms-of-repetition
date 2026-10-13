#!/usr/bin/env python3
"""Statistics and figures for teacher_forced_ablation.py outputs.

Unit of analysis is the source text (n = 500); Natural and ICL rows share sources, so the
condition contrast is paired. Per head and region:
  effect     mean change per condition, 95% CI, one-sample t-test, BH across heads x conditions
  contrast   ICL - Natural on the same sources, 95% CI, paired t-test, BH across heads
Across heads (primary, one test per checkpoint and region):
  Spearman correlation between attention scores (induction, previous-token, ...) and effects,
  with a permutation p-value over heads.
Group ablations: effect against number of ablated heads, top-k sets vs size-matched random sets.
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CKPT_ORDER = ["step0", "step1", "step1000", "step36000", "step71000", "step143000"]
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9a9a9a", "#222222"


def bh(p, q=0.05):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.zeros(len(p), bool)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    passed = pv[order] <= q * np.arange(1, len(pv) + 1) / len(pv)
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(len(pv), bool)
    sig[order[:k]] = True
    out[ok] = sig
    return out


def summarize(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_1samp(x, 0).pvalue if se > 0 else 1.0
    return m, m - 1.96 * se, m + 1.96 * se, p, n


def head_effects(single, metric):
    rows = []
    for (ck, region, layer, head), g in single.groupby(["checkpoint", "region", "layer", "head"]):
        for cond, gc in g.groupby("cond"):
            m, lo, hi, p, n = summarize(gc[metric])
            rows.append(dict(checkpoint=ck, region=region, layer=layer, head=head, cond=cond, mean=m, lo=lo, hi=hi, p=p, n=n))
        w = g.pivot_table(index="source", columns="cond", values=metric).dropna()
        if {"icl", "natural"} <= set(w.columns):
            m, lo, hi, p, n = summarize(w.icl - w.natural)
            rows.append(dict(checkpoint=ck, region=region, layer=layer, head=head, cond="icl_minus_natural", mean=m, lo=lo, hi=hi, p=p, n=n))
    df = pd.DataFrame(rows)
    df["sig_bh"] = False
    for (ck, region, is_contrast), idx in df.groupby(["checkpoint", "region", df.cond == "icl_minus_natural"]).groups.items():
        df.loc[idx, "sig_bh"] = bh(df.loc[idx, "p"].values)
    return df


def score_correlations(eff, scores, n_perm, rng):
    rows = []
    for (ck, region, cond), g in eff.groupby(["checkpoint", "region", "cond"]):
        s = scores[scores.checkpoint == ck]
        g = g.merge(s, on=["layer", "head"]).dropna(subset=["mean"])
        if len(g) < 10:
            continue
        for score in ["induction", "duplicate", "prev_token", "first_token"]:
            rho = stats.spearmanr(g[score], g["mean"]).correlation
            ranks_x, ranks_y = stats.rankdata(g[score]), stats.rankdata(g["mean"])
            null = np.array([np.corrcoef(ranks_x, rng.permutation(ranks_y))[0, 1] for _ in range(n_perm)])
            p = (1 + np.sum(np.abs(null) >= abs(rho))) / (n_perm + 1)
            rows.append(dict(checkpoint=ck, region=region, cond=cond, score=score, spearman=rho, p_perm=p, n_heads=len(g)))
    return pd.DataFrame(rows)


def group_summary(group, metric):
    rows = []
    for (ck, region, cond, kind, k, rep), g in group.groupby(["checkpoint", "region", "cond", "set", "k", "rep"]):
        m, lo, hi, p, n = summarize(g[metric])
        rows.append(dict(checkpoint=ck, region=region, cond=cond, set=kind, k=k, rep=rep, mean=m, lo=lo, hi=hi, p=p, n=n))
    return pd.DataFrame(rows)


def plot_group(gs, metric, region, out):
    cks = [c for c in CKPT_ORDER if c in set(gs.checkpoint)]
    fig, axes = plt.subplots(1, len(cks), figsize=(2.6 * len(cks), 3.0), sharey=True, squeeze=False)
    for ax, ck in zip(axes[0], cks):
        g = gs[(gs.checkpoint == ck) & (gs.region == region)]
        for cond, color in [("icl", BLUE), ("natural", ORANGE)]:
            gc = g[g.cond == cond]
            for kind, ls in [("top_induction", "-"), ("top_prev_token", ":")]:
                t = gc[gc.set == kind].sort_values("k")
                ax.plot(t.k, t["mean"], ls=ls, color=color, lw=2, marker="o", ms=4)
            r = gc[gc.set == "random"].groupby("k")["mean"].agg(["mean", "min", "max"]).reset_index()
            ax.fill_between(r.k, r["min"], r["max"], color=color, alpha=0.15, lw=0)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_title(ck, fontsize=9)
        ax.set_xlabel("heads ablated (k)")
    axes[0][0].set_ylabel(f"Δ {metric.replace('d_', '')} ({region})")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, label="ICL"), plt.Line2D([], [], color=ORANGE, lw=2, label="Natural"),
               plt.Line2D([], [], color=INK, lw=2, ls="-", label="top-k induction"), plt.Line2D([], [], color=INK, lw=2, ls=":", label="top-k prev-token"),
               plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.3, label="random k (range)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_head_scatter(eff, region, out):
    cks = [c for c in CKPT_ORDER if c in set(eff.checkpoint)]
    w = eff[(eff.region == region) & eff.cond.isin(["icl", "natural"])].pivot_table(
        index=["checkpoint", "layer", "head"], columns="cond", values=["mean", "sig_bh"], aggfunc="first")
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), squeeze=False)
    for ax, ck in zip(axes.flat, cks):
        g = w.loc[ck]
        x, y = g[("mean", "natural")], g[("mean", "icl")]
        sig = g[("sig_bh", "natural")].astype(bool) | g[("sig_bh", "icl")].astype(bool)
        lim = float(np.nanmax(np.abs(np.r_[x, y]))) * 1.1 or 1
        ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls="--")
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(0, color=INK, lw=0.6)
        ax.scatter(x[~sig], y[~sig], s=8, color=GREY, lw=0)
        ax.scatter(x[sig], y[sig], s=16, color=BLUE, lw=0)
        for (layer, head), xi, yi in zip(g.index, x, y):
            if np.hypot(xi, yi) > 0.5 * lim:
                ax.annotate(f"L{layer}H{head}", (xi, yi), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{ck}  ({int(sig.sum())} BH-sig.)", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Δ log p, Natural")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ log p, ICL")
    fig.suptitle(f"Single-head ablation, teacher-forced, region = {region}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default="outputs_teacher_forced/pythia14b")
    ap.add_argument("--ablation", default="zero")
    ap.add_argument("--metric", default="d_logprob", choices=["d_logprob", "d_margin", "d_flip"])
    ap.add_argument("--n_perm", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.input_dir)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    scores = pd.concat([pd.read_csv(f) for f in glob.glob(str(root / "*" / "head_scores.csv"))], ignore_index=True)
    scores.to_csv(out / "head_scores_all.csv", index=False)

    singles = glob.glob(str(root / "*" / f"single_{args.ablation}_L*.csv.gz"))
    if singles:
        single = pd.concat([pd.read_csv(f) for f in singles], ignore_index=True)
        eff = head_effects(single, args.metric)
        eff.to_csv(out / f"head_effects_{args.ablation}_{args.metric}.csv", index=False)
        cor = score_correlations(eff, scores, args.n_perm, rng)
        cor.to_csv(out / f"score_correlations_{args.ablation}_{args.metric}.csv", index=False)
        print("\n=== BH-significant heads (per checkpoint, region, condition) ===")
        print(eff[eff.sig_bh].groupby(["checkpoint", "region", "cond"]).size().unstack(fill_value=0).to_string())
        print("\n=== Spearman(score, effect) across heads, permutation p ===")
        print(cor[cor.score.isin(["induction", "prev_token"])].pivot_table(
            index=["checkpoint", "region", "score"], columns="cond", values="spearman").round(2).to_string())
        for region in ["cycle", "prompt2", "pre"]:
            plot_head_scatter(eff, region, out / f"head_scatter_{args.ablation}_{region}.png")

    groups = glob.glob(str(root / "*" / f"group_{args.ablation}.csv.gz"))
    if groups:
        group = pd.concat([pd.read_csv(f) for f in groups], ignore_index=True)
        gs = group_summary(group, args.metric)
        gs.to_csv(out / f"group_summary_{args.ablation}_{args.metric}.csv", index=False)
        for region in ["cycle", "prompt2", "pre"]:
            plot_group(gs, args.metric, region, out / f"group_curves_{args.ablation}_{region}.png")
        print("\n=== Group ablation, cycle region (mean over sources) ===")
        t = gs[(gs.region == "cycle") & (gs.rep == 0) & (gs.set != "random")]
        print(t.pivot_table(index=["checkpoint", "set", "k"], columns="cond", values="mean").round(3).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
