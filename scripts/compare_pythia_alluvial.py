#!/usr/bin/env python3
"""Analyze minipile alluvial results and compare with table results."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load minipile alluvial results
alluvial_csv = Path("outputs/EleutherAI/pythia-1.4b/minipile_alluvial/pile_natural_results.csv")
alluvial_df = pd.read_csv(alluvial_csv)

# Compute statistics
n_total = len(alluvial_df)
n_cyclic = int((alluvial_df['is_cyclical'] == True).sum())
pct_cyclic = 100.0 * n_cyclic / n_total if n_total > 0 else 0.0

has_cycle = alluvial_df[alluvial_df['is_cyclical'] == True]
mean_cycle_len = float(has_cycle['cycle_length'].mean()) if len(has_cycle) > 0 else np.nan
std_cycle_len = float(has_cycle['cycle_length'].std()) if len(has_cycle) > 0 else np.nan

# Load table results for comparison
table_csv = Path("plots/cycle_descriptive_stats.csv")
table_df = pd.read_csv(table_csv)
table_row = table_df[table_df['model'] == 'EleutherAI/pythia-1.4b']
if len(table_row) > 0:
    table_row = table_row.iloc[0]
else:
    table_row = None

# Build comparison markdown
lines = [
    "# Pythia-1.4B: Table vs Alluvial Comparison",
    "",
    "## Alluvial (Minipile, Natural)",
    f"- Model: EleutherAI/pythia-1.4b",
    f"- Dataset: Minipile (JeanKaddour/minipile)",
    f"- Generation: Greedy (do_sample=False)",
    f"- Prompts evaluated: {n_total}",
    f"- Cyclic sequences: {n_cyclic} ({pct_cyclic:.1f}%)",
    f"- Mean cycle length: {mean_cycle_len:.2f} ± {std_cycle_len:.2f} tokens",
    "",
]

if table_row is not None:
    lines.extend([
        "## Table (Minipile, Descriptive Stats)",
        f"- Model: EleutherAI/pythia-1.4b",
        f"- Dataset: Minipile (JeanKaddour/minipile)",
        f"- Generation: Greedy (do_sample=False)",
        f"- Prompts evaluated: {int(table_row['n_total'])}",
        f"- Cyclic sequences: {int(table_row['n_with_cycles'])} ({table_row['pct_cyclic']:.1f}%)",
        f"- Mean cycle length: {table_row['mean_cycle_length']:.2f} ± {table_row['std_cycle_length']:.2f} tokens",
        "",
    ])
    
    # Compare
    lines.extend([
        "## Comparison",
        f"- Cyclic %: Alluvial={pct_cyclic:.1f}% vs Table={table_row['pct_cyclic']:.1f}%",
        f"- Mean cycle length: Alluvial={mean_cycle_len:.2f} vs Table={table_row['mean_cycle_length']:.2f}",
        f"- Sample size ratio: Alluvial/Table = {n_total}/{int(table_row['n_total'])} = {n_total/table_row['n_total']:.1f}x",
        "",
        "**Note**: Alluvial uses `condition='natural'` and processes full sequences. Table uses truncated prompts (prompt_size=512) and distributed across ranks.",
    ])

summary_md = Path("plots/CYCLE_STATS_PYTHIA1_4B_COMPARISON.md")
summary_md.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved to {summary_md}")
