#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

plots = Path('plots')
agg = plots / 'cycle_descriptive_stats_aggregated.csv'
if not agg.exists():
    raise SystemExit('aggregated CSV not found')

df = pd.read_csv(agg)
# expected columns: model,n_total,n_with_cycles,pct_cyclic,mean_cycle_length,std_cycle_length,mean_cycle_count,std_cycle_count,latency

latex_lines = [
    r"\begin{tabular}{lrrrrr}",
    r"\textbf{Model} & \textbf{Cyclic Seq.} & \textbf{Pct. Repetitive} & \textbf{Cycle Size} & \textbf{Cycle Number} & \textbf{Latency} \\",
    r"\hline",
]
for _, row in df.iterrows():
    model = row['model']
    cyclic_seq = int(row['n_with_cycles'])
    pct_rep = row['pct_cyclic']
    cycle_size = row['mean_cycle_length']
    cycle_number = row['mean_cycle_count']
    latency = row['latency']
    latex_lines.append(f"{model} & {cyclic_seq} & {pct_rep:.1f} & {cycle_size:.2f} & {cycle_number:.2f} & {latency:.3f} \\")
latex_lines.append(r"\end{tabular}")
(plots / 'CYCLE_STATS_TABLE.tex').write_text('\n'.join(latex_lines))
print('Wrote', plots / 'CYCLE_STATS_TABLE.tex')
