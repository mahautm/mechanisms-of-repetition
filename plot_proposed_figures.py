#!/usr/bin/env python3
"""Candidate replacement figures for the reviewer round (written to plots/proposed/).

F1  per-head Δ_Natural vs Δ_ICL, one panel per checkpoint (replaces the S/D scatter)
F2  baseline repetition rate per checkpoint and condition, paired vs selected design (ceiling)
F3  cycle length and approximate onset distributions per model (replaces Table 1 means)
"""

import glob
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("plots/proposed")
CKPTS = ["step0", "step1", "step1000", "step36000", "step71000", "step143000"]
DESIGNS = {
    "paired": "outputs_ablation_paired_fp16/pythia14b/per_head_mcnemar_all.csv",
    "selected": "outputs_ablation_selected_fp16/pythia14b/per_head_mcnemar_all.csv",
}
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9a9a9a", "#222222"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def wilson(x, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_delta_scatter(name, path):
    df = pd.read_csv(path)
    wide = df.pivot_table(index=["checkpoint", "layer", "head"], columns="condition",
                          values=["delta", "sig_bh", "n"], aggfunc="first")
    lim = 0.10
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), sharex=True, sharey=True)
    for ax, ck in zip(axes.flat, CKPTS):
        if ck not in wide.index.get_level_values(0):
            ax.set_visible(False)
            continue
        w = wide.loc[ck]
        x, y = w[("delta", "natural")] * 100, w[("delta", "icl")] * 100
        sig = w[("sig_bh", "natural")].astype(bool) | w[("sig_bh", "icl")].astype(bool)
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(0, color=INK, lw=0.6)
        ax.plot([-lim * 100, lim * 100], [-lim * 100, lim * 100], color=GREY, lw=0.8, ls="--")
        ax.scatter(x[~sig], y[~sig], s=10, color=GREY, alpha=0.7, lw=0)
        ax.scatter(x[sig], y[sig], s=28, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
        for (layer, head), xi, yi in zip(w[sig].index, x[sig], y[sig]):
            ax.annotate(f"L{layer}H{head}", (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK)
        n_icl, n_nat = int(w[("n", "icl")].iloc[0]), int(w[("n", "natural")].iloc[0])
        clipped = int(((x.abs() > lim * 100) | (y.abs() > lim * 100)).sum())
        ax.set_title(f"{ck}  (n ICL={n_icl}, Nat={n_nat})" + (f"  [{clipped} off-axis]" if clipped else ""), fontsize=8.5)
        ax.set_xlim(-lim * 100, lim * 100)
        ax.set_ylim(-lim * 100, lim * 100)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("Δ repetition, Natural (pp)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ repetition, ICL (pp)")
    fig.suptitle(f"Single-head ablation effects, Pythia-1.4B, {name} prompts  (blue = BH-significant in either condition)", fontsize=10)
    fig.tight_layout()
    out = OUT / f"F1_delta_scatter_{name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_baseline_rates():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (name, path) in zip(axes, DESIGNS.items()):
        df = pd.read_csv(path).sort_values("n", ascending=False).drop_duplicates(["checkpoint", "condition"])
        for cond, color, off in [("icl", BLUE, -0.08), ("natural", ORANGE, 0.08)]:
            g = df[df.condition == cond].set_index("checkpoint").reindex(CKPTS)
            xs = np.arange(len(CKPTS)) + off
            rate = g.base_rate.values * 100
            ci = np.array([wilson(round(r * n), n) if n == n else (np.nan, np.nan) for r, n in zip(g.base_rate, g.n)]) * 100
            ax.errorbar(xs, rate, yerr=[np.clip(rate - ci[:, 0], 0, None), np.clip(ci[:, 1] - rate, 0, None)], color=color, lw=2, marker="o", ms=5,
                        capsize=0, elinewidth=1.2, label="ICL" if cond == "icl" else "Natural")
            for xi, ni in zip(xs, g.n.values):
                if ni == ni and ni < 50:
                    ax.annotate(f"n={int(ni)}", (xi, 60), ha="center", fontsize=7, color=color)
        ax.set_xticks(range(len(CKPTS)), [c.replace("step", "") for c in CKPTS])
        ax.set_xlabel("checkpoint (training step)")
        ax.set_title(f"{name} prompts", fontsize=9)
        ax.set_ylim(55, 101)
    axes[0].set_ylabel("baseline repetition (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Baseline repetition with 95% Wilson CIs: selected Natural prompts sit at the ceiling", fontsize=10)
    fig.tight_layout()
    out = OUT / "F2_baseline_rates_paired_vs_selected.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_cycle_stats():
    files = [f for f in glob.glob("plots/cycle_detection_*_rank*.csv") if "_step" not in f]
    groups = {}
    for f in files:
        groups.setdefault(re.sub(r"_rank\d+\.csv", "", f.split("cycle_detection_")[1]), []).append(f)
    pretty = {"EleutherAI_pythia-70m": "Pythia-70M", "EleutherAI_pythia-1_4b": "Pythia-1.4B",
              "EleutherAI_pythia-6_9b": "Pythia-6.9B", "allenai_OLMo-1B-hf": "OLMo-1B",
              "meta-llama_Llama-3_2-1B": "Llama-3.2-1B", "allenai_OLMo-2-1124-7B": "OLMo-2-1124-7B",
              "swiss-ai_Apertus-8B-2509": "Apertus-8B"}
    data = {pretty[k]: pd.concat([pd.read_csv(f, usecols=["cycle_length", "cycle_count"]) for f in v])
            for k, v in groups.items() if k in pretty}
    order = [m for m in pretty.values() if m in data]
    onset = [1000 - data[m].cycle_length * data[m].cycle_count for m in order]
    length = [data[m].cycle_length for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, vals, label in [(axes[0], onset, "approx. onset of repetition (tokens, 1000 − size × count)"),
                            (axes[1], length, "cycle length (tokens, log scale)")]:
        ax.boxplot(vals, vert=False, labels=order, showfliers=False, widths=0.55,
                   medianprops={"color": BLUE, "lw": 2}, boxprops={"color": INK, "lw": 0.8},
                   whiskerprops={"color": INK, "lw": 0.8}, capprops={"color": INK, "lw": 0.8})
        ax.set_xlabel(label)
    axes[1].set_xscale("log")
    axes[0].invert_yaxis()
    fig.suptitle("Greedy 1000-token generations, 1000 MiniPile prompts per model (every generation repeats)", fontsize=10)
    fig.tight_layout()
    out = OUT / "F3_cycle_stats_by_model.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in DESIGNS.items():
        print(fig_delta_scatter(name, path))
    print(fig_baseline_rates())
    print(fig_cycle_stats())
