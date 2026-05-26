#!/usr/bin/env python3
"""Create cross-model checkpoint comparison panels for Natural vs ICL."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def checkpoint_sort_key(checkpoint: str):
    match = re.search(r"step(\d+)", str(checkpoint))
    if match:
        return int(match.group(1)), str(checkpoint)
    if str(checkpoint) == "steplatest":
        return 10**12, str(checkpoint)
    return 10**12 - 1, str(checkpoint)


def ordered_checkpoints(df: pd.DataFrame):
    values = df["checkpoint"].dropna().astype(str).unique().tolist()
    return sorted(values, key=checkpoint_sort_key)


def normalize_checkpoint_order(checkpoints):
    # Keep numeric ordering, but explicitly pin step1 first if present.
    out = [str(c) for c in checkpoints]
    if "step1" in out:
        out = ["step1"] + [c for c in out if c != "step1"]
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = normalize_checkpoint_order(ordered_checkpoints(df))
    out = (
        df.groupby(["checkpoint", "condition"], as_index=False)
        .agg(
            mean_delta_rep_rate=("delta_repetition_rate", "mean"),
            std_delta_rep_rate=("delta_repetition_rate", "std"),
            mean_delta_cycle_count=("delta_avg_cycle_count", "mean"),
            std_delta_cycle_count=("delta_avg_cycle_count", "std"),
            mean_abs_delta_rep_rate=("delta_repetition_rate", lambda x: np.mean(np.abs(x))),
            std_abs_delta_rep_rate=("delta_repetition_rate", lambda x: np.std(np.abs(x), ddof=1) if len(x) > 1 else 0.0),
            n=("delta_repetition_rate", "count"),
        )
    )
    out["std_delta_rep_rate"] = out["std_delta_rep_rate"].fillna(0.0)
    out["std_delta_cycle_count"] = out["std_delta_cycle_count"].fillna(0.0)
    out["sem_delta_rep_rate"] = out["std_delta_rep_rate"] / np.sqrt(np.maximum(out["n"], 1))
    out["sem_delta_cycle_count"] = out["std_delta_cycle_count"] / np.sqrt(np.maximum(out["n"], 1))
    out["sem_abs_delta_rep_rate"] = out["std_abs_delta_rep_rate"] / np.sqrt(np.maximum(out["n"], 1))
    out["checkpoint"] = pd.Categorical(out["checkpoint"], categories=checkpoints, ordered=True)
    out = out.sort_values(["checkpoint", "condition"]).reset_index(drop=True)
    return out


def compute_row_ylim(model_data, metric_col: str, sem_col: str, ci_scale: float = 1.96):
    low_vals = []
    high_vals = []
    for summary in model_data.values():
        for condition in ["natural", "icl"]:
            sub = summary[summary["condition"] == condition]
            y = sub[metric_col].to_numpy(dtype=float)
            sem = sub[sem_col].to_numpy(dtype=float)
            low_vals.extend((y - ci_scale * sem).tolist())
            high_vals.extend((y + ci_scale * sem).tolist())
    low = float(np.nanmin(low_vals)) if low_vals else -1.0
    high = float(np.nanmax(high_vals)) if high_vals else 1.0
    pad = max((high - low) * 0.1, 1e-3)
    return low - pad, high + pad


def plot_panel_with_ci(model_data, output_file: Path) -> None:
    model_items = list(model_data.items())
    n_models = len(model_items)
    fig, axes = plt.subplots(2, n_models, figsize=(4.6 * n_models, 9.6), constrained_layout=False)
    axes = np.array(axes)
    if n_models == 1:
        axes = axes.reshape(2, 1)

    metric_map = [
        ("mean_delta_rep_rate", "sem_delta_rep_rate", "Mean Delta repetition rate"),
        ("mean_delta_cycle_count", "sem_delta_cycle_count", "Mean Delta cycle count"),
    ]

    line_colors = {"natural": "#2b83ba", "icl": "#d7191c"}
    line_styles = {"natural": "-", "icl": "--"}

    row_limits = [
        compute_row_ylim(model_data, metric_map[0][0], metric_map[0][1]),
        compute_row_ylim(model_data, metric_map[1][0], metric_map[1][1]),
    ]

    for col, (model_name, summary) in enumerate(model_items):
        checkpoints = summary["checkpoint"].cat.categories.tolist()
        x = np.arange(len(checkpoints))

        for row, (metric_col, sem_col, metric_title) in enumerate(metric_map):
            ax = axes[row, col]
            ax.set_box_aspect(1.0)
            for condition in ["natural", "icl"]:
                sub = summary[summary["condition"] == condition]
                y = []
                sem_y = []
                for cp in checkpoints:
                    cp_rows = sub[sub["checkpoint"] == cp]
                    values = cp_rows[metric_col].to_numpy()
                    sem_values = cp_rows[sem_col].to_numpy()
                    y.append(values[0] if len(values) else np.nan)
                    sem_y.append(sem_values[0] if len(sem_values) else np.nan)

                y = np.array(y, dtype=float)
                sem_y = np.array(sem_y, dtype=float)
                ci = 1.96 * sem_y

                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=4,
                    linewidth=2.0,
                    linestyle=line_styles[condition],
                    color=line_colors[condition],
                    label=condition.capitalize(),
                )
                ax.fill_between(x, y - ci, y + ci, color=line_colors[condition], alpha=0.18)

            ax.axhline(0, color="#444", linewidth=0.9)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(*row_limits[row])

            if row == 0:
                ax.set_title(model_name, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{metric_title}\n(95% CI)", fontsize=10)
            if row == 1:
                ax.set_xlabel("Checkpoint", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=True)
    fig.suptitle("Checkpoint Dynamics (Standardized Y by Row): Natural vs ICL", fontsize=14, y=1.03)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_abs_rep_panel_with_ci(model_data, output_file: Path) -> None:
    model_items = list(model_data.items())
    n_models = len(model_items)
    fig, axes = plt.subplots(1, n_models, figsize=(4.6 * n_models, 5.1), constrained_layout=False)
    axes = np.array(axes)
    if n_models == 1:
        axes = np.array([axes.item()])
    line_colors = {"natural": "#2b83ba", "icl": "#d7191c"}
    line_styles = {"natural": "-", "icl": "--"}

    y_min, y_max = compute_row_ylim(model_data, "mean_abs_delta_rep_rate", "sem_abs_delta_rep_rate")

    for col, (model_name, summary) in enumerate(model_items):
        ax = axes[col]
        ax.set_box_aspect(1.0)
        checkpoints = summary["checkpoint"].cat.categories.tolist()
        x = np.arange(len(checkpoints))

        for condition in ["natural", "icl"]:
            sub = summary[summary["condition"] == condition]
            y = []
            sem_y = []
            for cp in checkpoints:
                cp_rows = sub[sub["checkpoint"] == cp]
                values = cp_rows["mean_abs_delta_rep_rate"].to_numpy()
                sem_values = cp_rows["sem_abs_delta_rep_rate"].to_numpy()
                y.append(values[0] if len(values) else np.nan)
                sem_y.append(sem_values[0] if len(sem_values) else np.nan)

            y = np.array(y, dtype=float)
            sem_y = np.array(sem_y, dtype=float)
            ci = 1.96 * sem_y

            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=2.0,
                linestyle=line_styles[condition],
                color=line_colors[condition],
                label=condition.capitalize(),
            )
            ax.fill_between(x, y - ci, y + ci, color=line_colors[condition], alpha=0.18)

        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(y_min, y_max)
        ax.set_title(model_name, fontsize=12)
        ax.set_xlabel("Checkpoint", fontsize=10)
        if col == 0:
            ax.set_ylabel("Mean |Delta repetition rate|\n(95% CI)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=True)
    fig.suptitle("Absolute Repetition Effect Strength (Standardized Y): Natural vs ICL", fontsize=14, y=1.04)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    targets = {
        "Pythia-70m": Path("/home/mmahaut/projects/parrots/plots/ablation_pythia70m/combined_ablation_effects.csv"),
        "Pythia-1.4b": Path("/home/mmahaut/projects/parrots/plots/ablation_pythia14b/combined_ablation_effects.csv"),
        "OLMo-1B": Path("/home/mmahaut/projects/parrots/plots/ablation_olmo1b/combined_ablation_effects.csv"),
        "Apertus-8B": Path("/home/mmahaut/projects/parrots/plots/ablation_apertus8b/combined_ablation_effects.csv"),
    }

    model_data = {}
    for model_name, csv_path in targets.items():
        if not csv_path.exists():
            print(f"Skip {model_name}: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        model_data[model_name] = summarize(df)

    if not model_data:
        raise FileNotFoundError("No available combined_ablation_effects.csv files found")

    output_file = Path("/home/mmahaut/projects/parrots/plots/checkpoint_comparison_panel.png")
    output_file_abs = Path("/home/mmahaut/projects/parrots/plots/checkpoint_abs_rep_panel.png")
    plot_panel_with_ci(model_data, output_file)
    plot_abs_rep_panel_with_ci(model_data, output_file_abs)
    print(f"Saved {output_file}")
    print(f"Saved {output_file_abs}")


if __name__ == "__main__":
    main()
