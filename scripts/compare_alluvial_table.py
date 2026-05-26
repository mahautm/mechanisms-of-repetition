#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

plots = Path('plots')
agg_csv = plots / 'cycle_descriptive_stats_aggregated.csv'
if not agg_csv.exists():
    raise SystemExit('Aggregated CSV not found')
agg = pd.read_csv(agg_csv)
agg_idx = {r['model']: r for _, r in agg.iterrows()}

# find alluvial summaries
root = Path('outputs')
jsons = list(root.glob('**/minipile_alluvial/pile_eval_summary.json'))
if not jsons:
    print('No alluvial summaries found')
    raise SystemExit(0)

lines = ['# Alluvial vs Table Cyclic Rate Comparison', '']
lines.append('| Model | Alluvial N | Alluvial % | Table N | Table % | Delta (%) |')
lines.append('|---|---:|---:|---:|---:|---:|')

for js in sorted(jsons):
    data = json.loads(js.read_text())
    model_name = data.get('model_name')
    alluvial_n = data.get('natural', {}).get('n_samples', None)
    alluvial_pct = 100.0 * data.get('natural', {}).get('cyclical_rate', 0.0)
    table_row = agg_idx.get(model_name)
    table_n = int(table_row['n_total']) if table_row is not None else ''
    table_pct = float(table_row['pct_cyclic']) if table_row is not None else ''
    delta = (alluvial_pct - table_pct) if table_row is not None else ''
    lines.append(f'| {model_name} | {alluvial_n} | {alluvial_pct:.1f} | {table_n} | {table_pct:.1f} | {delta:.1f} |')

out = plots / 'CYCLE_STATS_ALLUVIAL_COMPARISON.md'
out.write_text('\n'.join(lines))
print('Wrote', out)
print('\n'.join(lines))
