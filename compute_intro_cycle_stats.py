#!/usr/bin/env python3
"""Compute descriptive cycle statistics for paper intro across model families."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\S+", text)


def detect_cycle_end(tokens: List[str], min_cycle: int = 2, max_cycle: int = 64) -> Tuple[int, int, int]:
    """Detect cycle that repeats until sequence end.

    Returns:
        (cycle_size, cycle_count, tokens_until_first_cycle)
    """
    n = len(tokens)
    if n < min_cycle * 2:
        return 0, 0, -1

    best = (0, 0, -1)
    max_size = min(max_cycle, n // 2)
    for size in range(min_cycle, max_size + 1):
        candidate = tokens[-size:]
        count = 0
        i = n - size
        while i >= 0 and tokens[i:i + size] == candidate:
            count += 1
            i -= size
        if count > 1:
            start = i + size
            # Choose first valid cycle with smallest size for consistency.
            return size, count, start
        if count == 1 and i == 0:
            start = i + size
            best = (size, count, start)
    return best


def clean_model_key_from_path(path: Path) -> str:
    parts = path.parts
    if "outputs" in parts:
        idx = parts.index("outputs")
        tail = parts[idx + 1 :]
        if len(tail) >= 2:
            return f"{tail[0]}/{tail[1]}"
        if len(tail) == 1:
            return tail[0]
    return str(path.parent)


def infer_family(model_key: str) -> str:
    lower = model_key.lower()
    if "pythia" in lower:
        return "pythia"
    if "llama" in lower:
        return "llama"
    if "olmo" in lower:
        return "olmo"
    if "apertus" in lower or "swiss-ai" in lower:
        return "apertus"
    if "opt" in lower or "gpt" in lower or "facebook/" in lower:
        return "gpt-open"
    return "other"


def load_cycle_files() -> List[Path]:
    root = Path("/home/mmahaut/projects/parrots/outputs")
    with_cycles = list(root.glob("**/slot_filling_results_with_cycles.csv"))
    raw = list(root.glob("**/slot_filling_results.csv"))

    # Keep raw only if paired with with_cycles is absent.
    with_cycles_parents = {p.parent for p in with_cycles}
    raw_only = [p for p in raw if p.parent not in with_cycles_parents]
    return sorted(with_cycles) + sorted(raw_only)


def stats_from_df(df: pd.DataFrame) -> Dict[str, float]:
    col_cycle_size = None
    if "cycle_size" in df.columns:
        col_cycle_size = "cycle_size"
    elif "cycle_length" in df.columns:
        col_cycle_size = "cycle_length"

    col_cycle_count = "cycle_count" if "cycle_count" in df.columns else None

    cycle_sizes = []
    cycle_counts = []
    first_cycle_tokens = []

    text_col: Optional[str] = None
    for c in ["generated", "generated_text"]:
        if c in df.columns:
            text_col = c
            break

    for _, row in df.iterrows():
        size = 0
        count = 0

        if col_cycle_size is not None and pd.notna(row[col_cycle_size]):
            try:
                size = int(row[col_cycle_size])
            except Exception:
                size = 0
        if col_cycle_count is not None and pd.notna(row[col_cycle_count]):
            try:
                count = int(row[col_cycle_count])
            except Exception:
                count = 0

        if text_col is not None:
            toks = tokenize_text(str(row[text_col]))
            d_size, d_count, d_first = detect_cycle_end(toks)
            if (size <= 0 or count <= 0) and d_count > 0:
                size, count = d_size, d_count
            if d_first >= 0:
                first_cycle_tokens.append(float(d_first))
            elif count > 0:
                first_cycle_tokens.append(np.nan)
        elif count > 0:
            first_cycle_tokens.append(np.nan)

        if count > 0 and size > 0:
            cycle_sizes.append(float(size))
            cycle_counts.append(float(count))

    n_total = len(df)
    n_cyclic = len(cycle_counts)

    return {
        "n_total": float(n_total),
        "n_cyclic": float(n_cyclic),
        "pct_cyclic": (100.0 * n_cyclic / n_total) if n_total else np.nan,
        "mean_cycle_size": float(np.mean(cycle_sizes)) if cycle_sizes else np.nan,
        "std_cycle_size": float(np.std(cycle_sizes, ddof=1)) if len(cycle_sizes) > 1 else np.nan,
        "mean_cycle_count": float(np.mean(cycle_counts)) if cycle_counts else np.nan,
        "std_cycle_count": float(np.std(cycle_counts, ddof=1)) if len(cycle_counts) > 1 else np.nan,
        "mean_tokens_to_first_cycle": float(np.nanmean(first_cycle_tokens)) if first_cycle_tokens else np.nan,
        "std_tokens_to_first_cycle": float(np.nanstd(first_cycle_tokens, ddof=1)) if len(first_cycle_tokens) > 1 else np.nan,
    }


def aggregate_family(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for family, g in rows.groupby("family"):
        n_total = g["n_total"].sum()
        n_cyclic = g["n_cyclic"].sum()
        out.append(
            {
                "family": family,
                "n_models": g["model_key"].nunique(),
                "n_total": n_total,
                "n_cyclic": n_cyclic,
                "pct_cyclic": (100.0 * n_cyclic / n_total) if n_total else np.nan,
                "mean_cycle_size": np.average(g["mean_cycle_size"].fillna(0), weights=np.maximum(g["n_cyclic"], 1)),
                "mean_cycle_count": np.average(g["mean_cycle_count"].fillna(0), weights=np.maximum(g["n_cyclic"], 1)),
                "mean_tokens_to_first_cycle": np.average(g["mean_tokens_to_first_cycle"].fillna(0), weights=np.maximum(g["n_cyclic"], 1)),
            }
        )
    return pd.DataFrame(out).sort_values("family")


def main() -> None:
    files = load_cycle_files()
    if not files:
        print("No slot-filling files found in outputs/.")
        return

    rows = []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        if df.empty:
            continue

        model_key = clean_model_key_from_path(file_path)
        family = infer_family(model_key)
        s = stats_from_df(df)
        s.update(
            {
                "model_key": model_key,
                "family": family,
                "source_file": str(file_path),
            }
        )
        rows.append(s)

    if not rows:
        print("No readable files with cycle-compatible schema.")
        return

    model_df = pd.DataFrame(rows).sort_values(["family", "model_key"])
    family_df = aggregate_family(model_df)

    out_dir = Path("/home/mmahaut/projects/parrots/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_csv = out_dir / "cycle_intro_stats_by_model.csv"
    family_csv = out_dir / "cycle_intro_stats_by_family.csv"

    model_df.to_csv(model_csv, index=False)
    family_df.to_csv(family_csv, index=False)

    print(f"Saved {model_csv}")
    print(f"Saved {family_csv}")
    print("\nFamily summary:")
    print(family_df.to_string(index=False))


if __name__ == "__main__":
    main()
