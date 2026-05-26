#!/usr/bin/env python3
"""Build paper-style checkpoint scatter summary figures directly from CSV data.

Layout:
- Top row: earlier checkpoints (small square panels)
- Bottom row: final checkpoint (large square panel spanning full width)
- One external legend per figure
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

def checkpoint_sort_key(checkpoint: str) -> Tuple[int, str]:
    match = re.search(r"step(\d+)", str(checkpoint))
    if match:
        return int(match.group(1)), str(checkpoint)
    if str(checkpoint) == "steplatest":
        return 10**12, str(checkpoint)
    return 10**12 - 1, str(checkpoint)


def nice_tick_step(limit: float) -> float:
    """Pick a human-friendly symmetric tick step for [-limit, limit]."""
    target = max(limit / 2.0, 0.01)
    candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for step in candidates:
        if step >= target:
            return step
    return candidates[-1]


def rounded_axis_limit(raw_limit: float) -> float:
    """Round axis limit up to a clean value so ticks are readable."""
    raw = max(0.02, float(raw_limit))
    step = nice_tick_step(raw)
    return math.ceil(raw / step) * step


def model_name_from_dir(dirname: str) -> str:
    aliases = {
        "ablation_pythia70m": "Pythia-70m",
        "ablation_pythia14b": "Pythia-1.4b",
        "ablation_olmo1b": "OLMo-1B",
        "ablation_apertus8b": "Apertus-8B",
    }
    if dirname in aliases:
        return aliases[dirname]
    if dirname.startswith("ablation_"):
        return dirname.replace("ablation_", "").replace("_", "-")
    return dirname


def layer_bucket(layer: int, max_layer: int) -> str:
    if max_layer <= 0:
        return "all"
    frac = layer / max_layer
    if frac <= 0.33:
        return "early"
    if frac <= 0.66:
        return "middle"
    return "late"


def style_category(shared: float, specific: float) -> str:
    # Category labels explicitly encode the signs of:
    # S = (ICL + Natural) / 2 and D = (ICL - Natural) / 2.
    if shared >= 0 and specific >= 0:
        return "S>0 Repetition-Suppressing + D>0 ICL-Boosted"
    if shared < 0 and specific >= 0:
        return "S<0 Repetition-Facilitating + D>0 ICL-Boosted"
    if shared >= 0 and specific < 0:
        return "S>0 Repetition-Suppressing + D<0 Natural-Boosted"
    return "S<0 Repetition-Facilitating + D<0 Natural-Boosted"


def panel_dataframe(df: pd.DataFrame, checkpoint: str, metric: str, jitter: float = 0.002) -> pd.DataFrame:
    sub = df[df["checkpoint"].astype(str) == checkpoint].copy()
    if sub.empty:
        raise ValueError(f"No rows for checkpoint {checkpoint}")

    pivot = sub.pivot_table(
        index=["layer", "head"],
        columns="condition",
        values=metric,
        aggfunc="mean",
    ).reset_index()
    if "icl" not in pivot.columns or "natural" not in pivot.columns:
        raise ValueError("Both icl and natural conditions are required")

    pivot["shared"] = 0.5 * (pivot["icl"] + pivot["natural"])
    pivot["specific"] = 0.5 * (pivot["icl"] - pivot["natural"])
    pivot["label"] = pivot.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)
    pivot["max_layer"] = int(pivot["layer"].max())
    pivot["layer_bucket"] = pivot.apply(lambda r: layer_bucket(int(r["layer"]), int(r["max_layer"])), axis=1)
    pivot["category"] = pivot.apply(lambda r: style_category(float(r["shared"]), float(r["specific"])), axis=1)
    pivot["magnitude"] = np.hypot(pivot["shared"], pivot["specific"])

    if jitter > 0:
        seed = sum(ord(c) for c in checkpoint) % (2**32)
        rng = np.random.default_rng(seed)
        pivot["shared_plot"] = pivot["shared"] + rng.uniform(-jitter, jitter, size=len(pivot))
        pivot["specific_plot"] = pivot["specific"] + rng.uniform(-jitter, jitter, size=len(pivot))
    else:
        pivot["shared_plot"] = pivot["shared"]
        pivot["specific_plot"] = pivot["specific"]

    return pivot


def draw_panel(
    ax,
    panel: pd.DataFrame,
    checkpoint: str,
    show_axis_labels: bool,
    annotate: bool,
    title_size: int,
    point_scale: float,
    axis_limit: float,
    title_inside: bool = False,
) -> None:
    lim = max(0.02, float(axis_limit))

    ax.axvspan(0, lim, ymin=0.5, ymax=1.0, color="#e7f6ec", alpha=0.65, zorder=0)
    ax.axvspan(-lim, 0, ymin=0.5, ymax=1.0, color="#fff4d9", alpha=0.65, zorder=0)
    ax.axvspan(0, lim, ymin=0.0, ymax=0.5, color="#d9edf8", alpha=0.6, zorder=0)
    ax.axvspan(-lim, 0, ymin=0.0, ymax=0.5, color="#fde4dc", alpha=0.62, zorder=0)

    if len(panel) > 1:
        ax.hexbin(
            panel["shared_plot"],
            panel["specific_plot"],
            gridsize=26,
            extent=(-lim, lim, -lim, lim),
            cmap="Blues",
            mincnt=1,
            bins="log",
            alpha=0.42,
            linewidths=0.0,
            zorder=1,
        )

    palette = {
        "S>0 Repetition-Suppressing + D>0 ICL-Boosted": "#1b9e77",
        "S<0 Repetition-Facilitating + D>0 ICL-Boosted": "#e6ab02",
        "S>0 Repetition-Suppressing + D<0 Natural-Boosted": "#377eb8",
        "S<0 Repetition-Facilitating + D<0 Natural-Boosted": "#d95f02",
    }
    marker_map = {"early": "o", "middle": "s", "late": "^", "all": "o"}

    for bucket, bucket_df in panel.groupby("layer_bucket"):
        for category, sub_df in bucket_df.groupby("category"):
            size = (30 + 170 * (sub_df["magnitude"] / max(sub_df["magnitude"].max(), 1e-6))) * point_scale
            ax.scatter(
                sub_df["shared_plot"],
                sub_df["specific_plot"],
                s=size,
                c=palette[category],
                marker=marker_map.get(bucket, "o"),
                alpha=0.80,
                edgecolors="#222222",
                linewidths=0.5,
                zorder=2,
            )

    if annotate:
        for _, row in panel.nlargest(12, "magnitude").iterrows():
            ax.annotate(
                row["label"],
                (row["shared"], row["specific"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#1d1d1d",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "#666", "lw": 0.3, "alpha": 0.78},
                zorder=3,
            )

    ax.axhline(0, color="#2f2f2f", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(0, color="#2f2f2f", linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # Enforce identical tick positions on all panels so shared axes are explicit.
    tick_step = nice_tick_step(lim)
    tick_vals = np.arange(-lim, lim + 0.5 * tick_step, tick_step)
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_box_aspect(1.0)
    if title_inside:
        ax.text(
            0.5,
            0.985,
            checkpoint,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=title_size,
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.82},
        )
    else:
        ax.set_title(checkpoint, fontsize=title_size, pad=1)

    if show_axis_labels:
        ax.set_xlabel("S (Shared Effect) = (ICL + Natural) / 2", fontsize=10, labelpad=4)
        ax.set_ylabel("D (Condition-Specific Effect) = (ICL - Natural) / 2", fontsize=10)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Keep ticks visible so axis comparability is obvious on all subplots.
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=8,
        )


def build_montage(model_dir: Path, metric_suffix_value: str) -> Path:
    csv_path = model_dir / "combined_ablation_effects.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    metric = "delta_repetition_rate" if metric_suffix_value == "rep_rate" else "delta_avg_cycle_count"
    df = pd.read_csv(csv_path)
    checkpoints = sorted(df["checkpoint"].dropna().astype(str).unique().tolist(), key=checkpoint_sort_key)
    if len(checkpoints) < 2:
        raise ValueError(f"Need at least 2 checkpoints in {csv_path}")

    left_ckpts = checkpoints[:-1]
    final_ckpt = checkpoints[-1]
    left_ckpts = left_ckpts[:4]
    n_left_slots = 4

    panels = {ck: panel_dataframe(df, ck, metric, jitter=0.002) for ck in checkpoints}
    global_abs = 0.0
    for panel in panels.values():
        global_abs = max(
            global_abs,
            float(np.nanmax(np.abs(panel["shared"]))),
            float(np.nanmax(np.abs(panel["specific"]))),
        )
    axis_limit = rounded_axis_limit(global_abs * 1.15)

    # Geometry tuned for a 2x2 grid on the left and one larger panel on the right.
    fig = plt.figure(figsize=(12.8, 7.9))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.18],
        height_ratios=[1.0, 1.0],
        hspace=0.18,
        wspace=0.14,
    )

    for i, ck in enumerate(left_ckpts):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        draw_panel(
            ax,
            panels[ck],
            ck,
            show_axis_labels=False,
            annotate=False,
            title_size=12,
            point_scale=0.55,
            axis_limit=axis_limit,
            title_inside=False,
        )

    for i in range(len(left_ckpts), n_left_slots):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    ax_big = fig.add_subplot(gs[:, 2])
    draw_panel(
        ax_big,
        panels[final_ckpt],
        final_ckpt,
        show_axis_labels=True,
        annotate=True,
        title_size=16,
        point_scale=1.18,
        axis_limit=axis_limit,
        title_inside=True,
    )

    model_name = model_name_from_dir(model_dir.name)

    category_palette = [
        ("S>0 Repetition-Suppressing + D>0 ICL-Boosted", "#1b9e77"),
        ("S<0 Repetition-Facilitating + D>0 ICL-Boosted", "#e6ab02"),
        ("S>0 Repetition-Suppressing + D<0 Natural-Boosted", "#377eb8"),
        ("S<0 Repetition-Facilitating + D<0 Natural-Boosted", "#d95f02"),
    ]
    category_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=11, markerfacecolor=color, markeredgecolor="#222222", label=label)
        for label, color in category_palette
    ]
    layer_handles = [
        Line2D([0], [0], marker=marker, linestyle="", markersize=11, color="#444", label=label)
        for label, marker in [("Early layers", "o"), ("Middle layers", "s"), ("Late layers", "^")]
    ]
    fig.legend(
        handles=category_handles + layer_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=4,
        frameon=True,
        fontsize=11,
        title=f"Legend ({model_name})",
        title_fontsize=12,
    )

    fig.subplots_adjust(left=0.055, right=0.995, top=0.965, bottom=0.20)

    out = model_dir / f"checkpoint_scatter_summary_{metric_suffix_value}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine checkpoint scatters into one summary figure per model")
    parser.add_argument("--plots-root", type=str, default="/home/mmahaut/projects/parrots/plots")
    parser.add_argument("--metric", choices=["rep_rate", "cycle_count"], default="rep_rate")
    parser.add_argument("--include-apertus", action="store_true")
    args = parser.parse_args()

    plots_root = Path(args.plots_root)
    model_dirs = [
        plots_root / "ablation_pythia70m",
        plots_root / "ablation_pythia14b",
        plots_root / "ablation_olmo1b",
    ]
    if args.include_apertus:
        model_dirs.append(plots_root / "ablation_apertus8b")

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Skip missing dir: {model_dir}")
            continue
        try:
            out = build_montage(model_dir, args.metric)
            print(f"Saved {out}")
        except Exception as exc:
            print(f"Skip {model_dir}: {exc}")


if __name__ == "__main__":
    main()
