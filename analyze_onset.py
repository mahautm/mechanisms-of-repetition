#!/usr/bin/env python3
"""Statistics and figures for onset_ablation.py outputs.

Per checkpoint, head set and condition: repetition rate and onset (generated tokens before the
first full cycle; sequences without a cycle are censored at the generation length, so onset is a
restricted mean). Changes are paired per sequence against the unablated run ("none"), and the
ICL - Natural difference in those changes is paired by source.
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CKPT_ORDER = ["step0", "step1", "step1000", "step36000", "step71000", "step143000"]
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9a9a9a", "#222222"


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def paired_changes(df):
    base = df[df.set == "none"].set_index(["checkpoint", "cond", "source"])[["repeats", "onset"]]
    rows = []
    keys = ["checkpoint", "set", "rep", "k", "heads"]
    for key, g in df[df.set != "none"].groupby(keys):
        g = g.set_index(["checkpoint", "cond", "source"])
        d = (g[["repeats", "onset"]].astype(float) - base.loc[g.index].astype(float)).reset_index()
        out = dict(zip(keys, key))
        for cond, gc in d.groupby("cond"):
            for m in ["repeats", "onset"]:
                mean, half, n = ci(gc[m])
                out.update({f"d_{m}_{cond}": mean, f"d_{m}_{cond}_ci": half})
        w = d.pivot_table(index="source", columns="cond", values="onset").dropna()
        mean, half, n = ci(w.icl - w.natural)
        out.update({"d_onset_icl_minus_nat": mean, "d_onset_icl_minus_nat_ci": half, "n_sources": n})
        rows.append(out)
    return pd.DataFrame(rows)


def plot(ch, base, metric, out):
    cks = [c for c in CKPT_ORDER if c in set(ch.checkpoint)]
    fig, axes = plt.subplots(1, len(cks), figsize=(2.7 * len(cks), 3.1), sharey=True, squeeze=False)
    for ax, ck in zip(axes[0], cks):
        g = ch[ch.checkpoint == ck]
        for cond, color in [("icl", BLUE), ("natural", ORANGE)]:
            col = f"d_{metric}_{cond}"
            for kind, ls in [("top_induction", "-"), ("top_prev_token", ":")]:
                t = g[g.set == kind].sort_values("k")
                ax.errorbar(t.k, t[col], yerr=t[col + "_ci"], ls=ls, color=color, lw=2, marker="o", ms=4, capsize=0)
            r = g[g.set == "random"].groupby("k")[col].agg(["min", "max"]).reset_index()
            ax.fill_between(r.k, r["min"], r["max"], color=color, alpha=0.15, lw=0)
            l12 = g[g.set == "L12H0"]
            ax.scatter(l12.k, l12[col], marker="*", s=90, color=color, edgecolor=INK, lw=0.6, zorder=4)
        b = base[base.checkpoint == ck]
        ax.set_title(f"{ck}\nbase: ICL {b.loc[b.cond == 'icl', metric].iloc[0]:.0f}, Nat {b.loc[b.cond == 'natural', metric].iloc[0]:.0f}"
                     if metric == "onset" else f"{ck}\nbase: ICL {b.loc[b.cond == 'icl', metric].iloc[0]:.2f}, Nat {b.loc[b.cond == 'natural', metric].iloc[0]:.2f}", fontsize=8.5)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("heads ablated (k)")
    axes[0][0].set_ylabel("Δ onset (tokens)" if metric == "onset" else "Δ repetition rate")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, label="ICL"), plt.Line2D([], [], color=ORANGE, lw=2, label="Natural"),
               plt.Line2D([], [], color=INK, lw=2, label="top-k induction"), plt.Line2D([], [], color=INK, lw=2, ls=":", label="top-k prev-token"),
               plt.Line2D([], [], color=INK, marker="*", ls="", ms=9, label="L12H0 alone"),
               plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.3, label="random k (range)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", default="outputs_onset/pythia14b")
    ap.add_argument("--ablation", default="zero")
    args = ap.parse_args()
    root = Path(args.input_dir)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(root / "*" / f"onset_{args.ablation}_*.csv.gz"))], ignore_index=True)
    base = df[df.set == "none"].groupby(["checkpoint", "cond"]).agg(repeats=("repeats", "mean"), onset=("onset", "mean")).reset_index()
    ch = paired_changes(df)
    base.to_csv(out / "baseline_rates.csv", index=False)
    ch.to_csv(out / f"paired_changes_{args.ablation}.csv", index=False)
    for metric in ["onset", "repeats"]:
        plot(ch, base, metric, out / f"onset_curves_{args.ablation}_{metric}.png")
    pd.set_option("display.width", 250)
    print(base.pivot_table(index="checkpoint", columns="cond", values=["repeats", "onset"]).reindex(CKPT_ORDER).dropna(how="all").round(3).to_string())
    cols = ["d_repeats_icl", "d_repeats_natural", "d_onset_icl", "d_onset_natural", "d_onset_icl_minus_nat", "d_onset_icl_minus_nat_ci"]
    t = ch[ch.set != "random"].sort_values(["checkpoint", "set", "k"])
    print(t[["checkpoint", "set", "k"] + cols].round(3).to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
