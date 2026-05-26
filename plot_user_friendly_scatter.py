#!/usr/bin/env python3
"""Create user-friendly shared-vs-specific scatter plots from combined ablation CSV files."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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


def discover_targets(plots_root: Path) -> List[Dict[str, str]]:
    targets: List[Dict[str, str]] = []
    for csv_path in sorted(plots_root.glob("ablation_*/combined_ablation_effects.csv")):
        model_dir = csv_path.parent
        model_name = model_name_from_dir(model_dir.name)
        targets.append(
            {
                "model": model_name,
                "csv": str(csv_path),
            }
        )
    return targets


def checkpoint_sort_key(checkpoint: str):
    match = re.search(r"step(\d+)", str(checkpoint))
    if match:
        return int(match.group(1)), str(checkpoint)
    if str(checkpoint) == "steplatest":
        return 10**12, str(checkpoint)
    return 10**12 - 1, str(checkpoint)


def checkpoint_slug(checkpoint: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(checkpoint))


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
    # Category labels explicitly encode signs of
    # S = (ICL + Natural) / 2 and D = (ICL - Natural) / 2.
    if shared >= 0 and specific >= 0:
        return "S>0 Repetition-Suppressing + D>0 ICL-Boosted"
    if shared < 0 and specific >= 0:
        return "S<0 Repetition-Facilitating + D>0 ICL-Boosted"
    if shared >= 0 and specific < 0:
        return "S>0 Repetition-Suppressing + D<0 Natural-Boosted"
    return "S<0 Repetition-Facilitating + D<0 Natural-Boosted"


def metric_label(metric: str) -> str:
    labels = {
        "delta_repetition_rate": "Delta repetition rate",
        "delta_avg_cycle_count": "Delta avg cycle count",
    }
    return labels.get(metric, metric)


def metric_suffix(metric: str) -> str:
    suffixes = {
        "delta_repetition_rate": "rep_rate",
        "delta_avg_cycle_count": "cycle_count",
    }
    return suffixes.get(metric, metric)


def friendly_scatter(
    input_csv: Path,
    output_png: Path,
    model_name: str,
    metric: str,
    jitter: float = 0.0,
    checkpoint: Optional[str] = None,
    show_legends: bool = True,
    show_title: bool = True,
) -> None:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"No rows in {input_csv}")

    checkpoints = sorted(df["checkpoint"].dropna().astype(str).unique().tolist(), key=checkpoint_sort_key)
    if not checkpoints:
        raise ValueError("No checkpoints found")
    latest = checkpoints[-1]
    selected = latest if checkpoint is None else str(checkpoint)
    if selected not in checkpoints:
        raise ValueError(f"Checkpoint {selected} not found in {input_csv}")

    latest_df = df[df["checkpoint"].astype(str) == selected].copy()
    if latest_df.empty:
        raise ValueError(f"No rows for checkpoint {selected}")

    pivot = latest_df.pivot_table(
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

    # Optional plotting jitter helps visualize dense overlap while preserving true labels/selection.
    pivot["shared_plot"] = pivot["shared"]
    pivot["specific_plot"] = pivot["specific"]
    if jitter > 0:
        rng = np.random.default_rng(7)
        pivot["shared_plot"] = pivot["shared"] + rng.uniform(-jitter, jitter, size=len(pivot))
        pivot["specific_plot"] = pivot["specific"] + rng.uniform(-jitter, jitter, size=len(pivot))

    top_labels = pivot.nlargest(14, "magnitude")

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11, 8))

    # Soft quadrant shading for easier reading.
    xlim = max(0.02, float(np.nanmax(np.abs(pivot["shared"]))) * 1.15)
    ylim = max(0.02, float(np.nanmax(np.abs(pivot["specific"]))) * 1.15)
    ax.axvspan(0, xlim, ymin=0.5, ymax=1.0, color="#e7f6ec", alpha=0.65, zorder=0)
    ax.axvspan(-xlim, 0, ymin=0.5, ymax=1.0, color="#fff4d9", alpha=0.65, zorder=0)
    ax.axvspan(0, xlim, ymin=0.0, ymax=0.5, color="#d9edf8", alpha=0.6, zorder=0)
    ax.axvspan(-xlim, 0, ymin=0.0, ymax=0.5, color="#fde4dc", alpha=0.62, zorder=0)

    if len(pivot) > 1:
        ax.hexbin(
            pivot["shared_plot"],
            pivot["specific_plot"],
            gridsize=28,
            extent=(-xlim, xlim, -ylim, ylim),
            cmap="Blues",
            mincnt=1,
            bins="log",
            alpha=0.42,
            linewidths=0.0,
            zorder=1,
        )

    palette: Dict[str, str] = {
        "S>0 Repetition-Suppressing + D>0 ICL-Boosted": "#1b9e77",
        "S<0 Repetition-Facilitating + D>0 ICL-Boosted": "#e6ab02",
        "S>0 Repetition-Suppressing + D<0 Natural-Boosted": "#377eb8",
        "S<0 Repetition-Facilitating + D<0 Natural-Boosted": "#d95f02",
    }
    marker_map = {"early": "o", "middle": "s", "late": "^", "all": "o"}

    for bucket, bucket_df in pivot.groupby("layer_bucket"):
        for category, sub_df in bucket_df.groupby("category"):
            size = 45 + 220 * (sub_df["magnitude"] / max(sub_df["magnitude"].max(), 1e-6))
            ax.scatter(
                sub_df["shared_plot"],
                sub_df["specific_plot"],
                s=size,
                c=palette[category],
                marker=marker_map.get(bucket, "o"),
                alpha=0.78,
                edgecolors="#222222",
                linewidths=0.6,
                zorder=2,
            )

    for _, row in top_labels.iterrows():
        ax.annotate(
            row["label"],
            (row["shared"], row["specific"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#1d1d1d",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#555", "lw": 0.4, "alpha": 0.8},
            zorder=3,
        )

    ax.axhline(0, color="#2f2f2f", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(0, color="#2f2f2f", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    if show_title:
        ax.set_title(
            f"{model_name}: Shared vs Condition-Specific Head Effects\n"
            f"Metric: {metric_label(metric)} | Checkpoint: {selected}",
            fontsize=13,
            pad=12,
        )
    ax.set_xlabel("S (Shared Effect) = (ICL + Natural) / 2", fontsize=11)
    ax.set_ylabel("D (Condition-Specific Effect) = (ICL - Natural) / 2", fontsize=11)

    if show_legends:
        category_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="#222222", markersize=8, label=label)
            for label, color in palette.items()
        ]
        bucket_handles = [
            Line2D([0], [0], marker=marker, color="#444", linestyle="", markersize=8, label=f"{bucket.title()} layers")
            for bucket, marker in [("early", "o"), ("middle", "s"), ("late", "^")]
        ]

        leg1 = ax.legend(handles=category_handles, loc="upper left", fontsize=8, title="Effect category")
        ax.add_artist(leg1)
        ax.legend(handles=bucket_handles, loc="lower right", fontsize=8, title="Layer position")

    fig.tight_layout()
    fig.savefig(output_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create user-friendly shared-vs-specific scatter plots")
    parser.add_argument("--metric", default="delta_repetition_rate", choices=["delta_repetition_rate", "delta_avg_cycle_count"])
    parser.add_argument("--discover", action="store_true", help="Auto-discover ablation folders under plots/")
    parser.add_argument("--jitter", type=float, default=0.0, help="Uniform jitter radius for dense overlap (plotting only)")
    parser.add_argument("--all-checkpoints", action="store_true", help="Generate one plot per checkpoint")
    parser.add_argument("--checkpoints", type=str, default="", help="Comma-separated checkpoint names to render")
    parser.add_argument("--no-legends", action="store_true", help="Render plots without in-panel legends")
    args = parser.parse_args()

    plots_root = Path("/home/mmahaut/projects/parrots/plots")
    if args.discover:
        targets = discover_targets(plots_root)
    else:
        targets: List[Dict[str, str]] = [
            {
                "model": "Pythia-70m",
                "csv": "/home/mmahaut/projects/parrots/plots/ablation_pythia70m/combined_ablation_effects.csv",
            },
            {
                "model": "Pythia-1.4b",
                "csv": "/home/mmahaut/projects/parrots/plots/ablation_pythia14b/combined_ablation_effects.csv",
            },
            {
                "model": "OLMo-1B",
                "csv": "/home/mmahaut/projects/parrots/plots/ablation_olmo1b/combined_ablation_effects.csv",
            },
            {
                "model": "Apertus-8B",
                "csv": "/home/mmahaut/projects/parrots/plots/ablation_apertus8b/combined_ablation_effects.csv",
            },
        ]

    for item in targets:
        csv_path = Path(item["csv"])
        if not csv_path.exists():
            print(f"Skip {item['model']}: missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        checkpoints = sorted(df["checkpoint"].dropna().astype(str).unique().tolist(), key=checkpoint_sort_key)
        if not checkpoints:
            print(f"Skip {item['model']}: no checkpoints in {csv_path}")
            continue

        if args.checkpoints.strip():
            requested = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
            selected_checkpoints = [x for x in requested if x in checkpoints]
            missing = [x for x in requested if x not in checkpoints]
            if missing:
                print(f"Skip missing checkpoints for {item['model']}: {missing}")
        elif args.all_checkpoints:
            selected_checkpoints = checkpoints
        else:
            selected_checkpoints = [checkpoints[-1]]

        for ck in selected_checkpoints:
            slug = checkpoint_slug(ck)
            png_name = f"shared_vs_specific_scatter_{metric_suffix(args.metric)}_{slug}.png"
            if args.jitter > 0:
                png_name = f"shared_vs_specific_scatter_{metric_suffix(args.metric)}_{slug}_jitter.png"
            png_path = csv_path.parent / png_name
            friendly_scatter(
                csv_path,
                png_path,
                item["model"],
                args.metric,
                jitter=args.jitter,
                checkpoint=ck,
                show_legends=not args.no_legends,
            )
            print(f"Saved {png_path}")

            # Keep compatibility: also emit legacy latest-checkpoint filename.
            if ck == checkpoints[-1]:
                legacy_name = f"shared_vs_specific_scatter_{metric_suffix(args.metric)}.png"
                if args.jitter > 0:
                    legacy_name = f"shared_vs_specific_scatter_{metric_suffix(args.metric)}_jitter.png"
                legacy_path = csv_path.parent / legacy_name
                if legacy_path != png_path:
                    friendly_scatter(
                        csv_path,
                        legacy_path,
                        item["model"],
                        args.metric,
                        jitter=args.jitter,
                        checkpoint=ck,
                        show_legends=not args.no_legends,
                    )
                    print(f"Saved {legacy_path}")


if __name__ == "__main__":
    main()
