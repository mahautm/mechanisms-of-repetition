#!/usr/bin/env python3
import re
from pathlib import Path

import numpy as np
import pandas as pd

plots = Path('plots')
files = sorted([str(p) for p in plots.glob('cycle_detection_*.csv')])
print('Found', len(files), 'files')
if not files:
    raise SystemExit('No cycle_detection files found')

rows = []
for f in files:
    df = pd.read_csv(f)
    if df.empty:
        continue
    df['_src'] = f
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
# Normalize column types
all_df['cycle_count'] = pd.to_numeric(all_df['cycle_count'], errors='coerce').fillna(0).astype(int)
all_df['cycle_length'] = pd.to_numeric(all_df['cycle_length'], errors='coerce').fillna(0).astype(float)
all_df['elapsed_seconds'] = pd.to_numeric(all_df['elapsed_seconds'], errors='coerce').fillna(0).astype(float)

results = []
for model, g in all_df.groupby('model'):
    n_total = len(g)
    n_with = int((g['cycle_count'] > 0).sum())
    pct = 100.0 * n_with / n_total if n_total > 0 else 0.0
    has = g[g['cycle_count'] > 0]
    mean_len = float(has['cycle_length'].mean()) if not has.empty else float('nan')
    std_len = float(has['cycle_length'].std()) if not has.empty else float('nan')
    mean_count = float(has['cycle_count'].mean()) if not has.empty else float('nan')
    std_count = float(has['cycle_count'].std()) if not has.empty else float('nan')
    # compute latency as sum(file_elapsed_seconds)/total_examples
    src_groups = g.groupby('_src')
    total_elapsed = 0.0
    total_examples = 0
    for src, sg in src_groups:
        file_elapsed = float(sg['elapsed_seconds'].iloc[0]) if 'elapsed_seconds' in sg.columns else 0.0
        file_n = len(sg)
        total_elapsed += file_elapsed
        total_examples += file_n
    latency = float(total_elapsed / total_examples) if total_examples > 0 else float('nan')

    results.append({
        'model': model,
        'n_total': n_total,
        'n_with_cycles': n_with,
        'pct_cyclic': pct,
        'mean_cycle_length': mean_len,
        'std_cycle_length': std_len,
        'mean_cycle_count': mean_count,
        'std_cycle_count': std_count,
        'latency': latency,
    })

# helper: parse model size heuristically (for ordering)
def parse_model_size(model_name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)([kKmMgGbB])", model_name)
    if not m:
        return 0.0
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == 'k':
        return num * 1e3
    if unit == 'm':
        return num * 1e6
    if unit == 'g' or unit == 'b':
        return num * 1e9
    return num

out_df = pd.DataFrame(results)
out_df['model_size'] = out_df['model'].apply(parse_model_size)
# sort by model size desc, then model name
out_df = out_df.sort_values(['model_size', 'model'], ascending=[False, True]).drop(columns=['model_size'])

out_csv = plots / 'cycle_descriptive_stats_aggregated.csv'
out_csv2 = plots / 'cycle_descriptive_stats.csv'
out_df.to_csv(out_csv, index=False)
out_df.to_csv(out_csv2, index=False)
print('Wrote', out_csv)

# write markdown
summary_lines = ['# Cycle Descriptive Statistics Summary (aggregated)', '']
for _, row in out_df.iterrows():
    summary_lines.append(f"## {row['model']}")
    summary_lines.append(f"- Total examples: {int(row['n_total'])}")
    summary_lines.append(f"- Examples with cycles: {int(row['n_with_cycles'])} ({row['pct_cyclic']:.1f}%)")
    summary_lines.append(f"- Percentage repetitive sentences: {row['pct_cyclic']:.1f}%")
    summary_lines.append(f"- Mean cycle length: {row['mean_cycle_length']:.2f} ± {row['std_cycle_length']:.2f} tokens")
    summary_lines.append(f"- Mean cycle count: {row['mean_cycle_count']:.2f} ± {row['std_cycle_count']:.2f}")
    summary_lines.append('')
summary_text = '\n'.join(summary_lines)
(plots / 'CYCLE_STATS_SUMMARY.md').write_text(summary_text)
print('Wrote', plots / 'CYCLE_STATS_SUMMARY.md')

# write latex (no total repetitions column; include pct repetitive)
latex_lines = [
    r"\begin{tabular}{lrrrrr}",
    r"\textbf{Model} & \textbf{Cyclic Seq.} & \textbf{Pct. Repetitive} & \textbf{Cycle Size} & \textbf{Cycle Number} & \textbf{Latency} \\",
    r"\hline",
]
for _, row in out_df.iterrows():
    cyclic_seq = int(row['n_with_cycles'])
    pct_rep = row['pct_cyclic']
    cycle_size = row['mean_cycle_length']
    cycle_number = row['mean_cycle_count']
    latency = row['latency']
    latex_lines.append(f"{row['model']} & {cyclic_seq} & {pct_rep:.1f} & {cycle_size:.2f} & {cycle_number:.2f} & {latency:.3f} \\")
latex_lines.append(r"\end{tabular}")
(plots / 'CYCLE_STATS_TABLE.tex').write_text('\n'.join(latex_lines))
print('Wrote', plots / 'CYCLE_STATS_TABLE.tex')

print('\nAggregated summary:')
print(out_df.to_csv(index=False))
