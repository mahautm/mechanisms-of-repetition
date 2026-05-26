#!/usr/bin/env python3
"""Summarize per-model combined_ablation_effects.csv files.

Produces:
- plots/<model>/quadrant_timeseries.csv
- plots/<model>/delta_pairs_last.csv
- plots/<model>/delta_icl_vs_delta_natural_last.png

Usage: run from repo root: python tools/summarize_ablation_quadrants.py
"""
import csv
import glob
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def process_file(path):
    model_dir = os.path.dirname(path)
    model_name = os.path.basename(model_dir)

    # read rows
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for line in r:
            line['delta_repetition_rate'] = float(line['delta_repetition_rate'])
            line['head'] = int(line['head'])
            line['layer'] = int(line['layer'])
            rows.append(line)

    checkpoints = sorted({r['checkpoint'] for r in rows})
    if not checkpoints:
        print('No checkpoints in', path)
        return

    timeseries_rows = []
    for ck in checkpoints:
        # compute mean deltas per condition
        icl_vals = [r['delta_repetition_rate'] for r in rows if r['checkpoint'] == ck and r['condition'] == 'icl']
        nat_vals = [r['delta_repetition_rate'] for r in rows if r['checkpoint'] == ck and r['condition'] == 'natural']
        mean_icl = sum(icl_vals) / len(icl_vals) if icl_vals else 0.0
        mean_nat = sum(nat_vals) / len(nat_vals) if nat_vals else 0.0

        # quadrant counts
        pairs = defaultdict(dict)
        for r in rows:
            if r['checkpoint'] != ck:
                continue
            key = (r['head'], r['layer'])
            pairs[key][r['condition']] = r['delta_repetition_rate']

        counts = {'SposDpos': 0, 'SposDneg': 0, 'SnegDpos': 0, 'SnegDneg': 0}
        for v in pairs.values():
            if 'icl' not in v or 'natural' not in v:
                continue
            icl = v['icl']; nat = v['natural']
            S = 0.5 * (icl + nat)
            D = 0.5 * (icl - nat)
            if S > 0 and D > 0:
                counts['SposDpos'] += 1
            elif S > 0 and D < 0:
                counts['SposDneg'] += 1
            elif S < 0 and D > 0:
                counts['SnegDpos'] += 1
            elif S < 0 and D < 0:
                counts['SnegDneg'] += 1

        timeseries_rows.append({
            'checkpoint': ck,
            'mean_delta_icl': mean_icl,
            'mean_delta_natural': mean_nat,
            **counts,
        })

    # write timeseries CSV
    out_ts = os.path.join(model_dir, 'quadrant_timeseries.csv')
    with open(out_ts, 'w', newline='') as f:
        fieldnames = ['checkpoint', 'mean_delta_icl', 'mean_delta_natural', 'SposDpos', 'SposDneg', 'SnegDpos', 'SnegDneg']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in timeseries_rows:
            w.writerow(r)

    # last checkpoint per-head CSV and scatter PNG
    last = checkpoints[-1]
    pairs = defaultdict(dict)
    for r in rows:
        if r['checkpoint'] != last:
            continue
        key = (r['head'], r['layer'])
        pairs[key][r['condition']] = r['delta_repetition_rate']

    out_pairs = os.path.join(model_dir, 'delta_pairs_last.csv')
    with open(out_pairs, 'w', newline='') as f:
        fieldnames = ['head', 'layer', 'icl', 'natural', 'S', 'D']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        xs = []
        ys = []
        cols = []
        for (head, layer), v in sorted(pairs.items()):
            if 'icl' not in v or 'natural' not in v:
                continue
            icl = v['icl']; nat = v['natural']
            S = 0.5 * (icl + nat)
            D = 0.5 * (icl - nat)
            w.writerow({'head': head, 'layer': layer, 'icl': icl, 'natural': nat, 'S': S, 'D': D})
            xs.append(icl); ys.append(nat)
            if S > 0 and D > 0:
                cols.append('C0')
            elif S > 0 and D < 0:
                cols.append('C1')
            elif S < 0 and D > 0:
                cols.append('C2')
            else:
                cols.append('C3')

    out_png = os.path.join(model_dir, 'delta_icl_vs_delta_natural_last.png')
    if plt and xs:
        plt.figure(figsize=(5,5))
        plt.axline((0,0),(1,1), color='gray', linestyle='--', linewidth=1)
        plt.scatter(xs, ys, c=cols, s=20, alpha=0.8)
        plt.xlabel('delta_icl (ablated - baseline)')
        plt.ylabel('delta_natural (ablated - baseline)')
        plt.title(f'{model_name} last checkpoint: {last}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    else:
        if not plt:
            print('matplotlib not available; skipping PNG for', model_name)

    print('Wrote:', out_ts, out_pairs, out_png)


def main():
    files = glob.glob('plots/*/combined_ablation_effects.csv')
    if not files:
        print('No combined_ablation_effects.csv files found under plots/*/')
        return
    for p in sorted(files):
        process_file(p)


if __name__ == '__main__':
    main()
