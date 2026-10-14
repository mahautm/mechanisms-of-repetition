#!/usr/bin/env python3
"""Paper figures from the teacher-forced and onset experiments (writes to figures/<tag>/).

M1  emergence of induction and previous-token heads over training
M2  cycle-entry survival curves: share of prompts not yet repeating vs generated tokens,
    unablated and with the top-8 induction / previous-token heads ablated
M3  condition dissociation: ICL - Natural change in onset per head set and checkpoint
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CKPT_ORDER = ["step0", "step1", "step1000", "step36000", "step71000", "step143000"]
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9a9a", "#222222", "#666666"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def ckpt_key(c):
    return CKPT_ORDER.index(c) if c in CKPT_ORDER else int(c.split("-")[0].replace("step", ""))


def short(c):
    s = c.split("-")[0].replace("step", "")
    return f"{int(s) // 1000}k" if int(s) >= 1000 else s


def fig_emergence(tf_dir, out):
    s = pd.concat([pd.read_csv(f) for f in glob.glob(f"{tf_dir}/*/head_scores.csv")])
    cks = sorted(s.checkpoint.unique(), key=ckpt_key)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    for ax, score, title in [(axes[0], "induction", "Induction score"), (axes[1], "prev_token", "Previous-token score")]:
        top = [s[s.checkpoint == c][score].nlargest(8).values for c in cks]
        x = np.arange(len(cks))
        for i in range(8):
            ax.plot(x, [t[i] for t in top], color=BLUE if score == "induction" else ORANGE, lw=1.2, alpha=1 - i * 0.09, marker="o", ms=3)
        med = [s[s.checkpoint == c][score].median() for c in cks]
        ax.plot(x, med, color=MUTED, lw=1.2, ls="--", label="median head")
        ax.set_xticks(x, [short(c) for c in cks])
        ax.set_title(f"{title}: top 8 heads", fontsize=9)
        ax.set_xlabel("checkpoint")
    axes[0].set_ylabel("mean attention to target position")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def survival(onsets, horizon):
    t = np.arange(horizon + 1)
    return t, np.array([(onsets > x).mean() for x in t])


def fig_survival(on_dir, out, horizon=512):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{on_dir}/*/onset_zero_*.csv.gz")])
    cks = sorted(df.checkpoint.unique(), key=ckpt_key)
    fig, axes = plt.subplots(1, len(cks), figsize=(2.6 * len(cks), 2.9), sharey=True, squeeze=False)
    for ax, ck in zip(axes[0], cks):
        d = df[df.checkpoint == ck]
        for cond, color in [("icl", BLUE), ("natural", ORANGE)]:
            for sel, ls, lw in [(d.set == "none", "-", 2), ((d.set == "top_induction") & (d.k == 8), "--", 1.4),
                                ((d.set == "top_prev_token") & (d.k == 8), ":", 1.6)]:
                o = d[sel & (d.cond == cond)]
                onset = np.where(o.repeats, o.onset, np.inf)
                t, sv = survival(onset, horizon)
                ax.plot(t, sv, color=color, ls=ls, lw=lw)
        ax.set_title(short(ck), fontsize=9)
        ax.set_xlabel("generated tokens")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("share not yet in a cycle")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, label="ICL"), plt.Line2D([], [], color=ORANGE, lw=2, label="Natural"),
               plt.Line2D([], [], color=INK, lw=2, label="no ablation"), plt.Line2D([], [], color=INK, lw=1.4, ls="--", label="top-8 induction ablated"),
               plt.Line2D([], [], color=INK, lw=1.6, ls=":", label="top-8 previous-token ablated")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dissociation(on_dir, out, k=8):
    ch = pd.read_csv(f"{on_dir}/analysis/paired_changes_zero.csv")
    cks = [c for c in sorted(ch.checkpoint.unique(), key=ckpt_key)]
    x = np.arange(len(cks))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.axhline(0, color=INK, lw=0.6)
    rnd = ch[(ch.set == "random") & (ch.k == k)].groupby("checkpoint")["d_onset_icl_minus_nat"].agg(["min", "max"]).reindex(cks)
    ax.fill_between(x, rnd["min"], rnd["max"], color=GREY, alpha=0.3, lw=0, label=f"random {k} heads (range)")
    for kind, k_, color, label, off in [("top_induction", k, BLUE, f"top-{k} induction", -0.12),
                                        ("top_prev_token", k, ORANGE, f"top-{k} previous-token", 0.0),
                                        ("L12H0", 1, INK, "L12H0 alone", 0.12)]:
        t = ch[(ch.set == kind) & (ch.k == k_)].set_index("checkpoint").reindex(cks)
        if t["d_onset_icl_minus_nat"].notna().any():
            ax.errorbar(x + off, t["d_onset_icl_minus_nat"], yerr=t["d_onset_icl_minus_nat_ci"], color=color, marker="o", ms=5,
                        lw=0, elinewidth=1.4, capsize=0, label=label)
    ax.set_xticks(x, [short(c) for c in cks])
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("Δ onset ICL − Δ onset Natural\n(tokens, paired by source)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="pythia14b")
    args = ap.parse_args()
    out = Path("figures") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    tf_dir, on_dir = f"outputs_teacher_forced/{args.tag}", f"outputs_onset/{args.tag}"
    fig_emergence(tf_dir, out / "M1_head_emergence.png")
    fig_survival(on_dir, out / "M2_onset_survival.png")
    fig_dissociation(on_dir, out / "M3_onset_dissociation.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
