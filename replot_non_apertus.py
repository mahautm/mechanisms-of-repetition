#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

from plot_checkpoint_comparison_panel import (
    plot_abs_rep_panel_with_ci,
    plot_panel_with_ci,
    summarize,
)
from plot_user_friendly_scatter import (
    checkpoint_slug,
    checkpoint_sort_key,
    friendly_scatter,
    metric_suffix,
)


def main() -> None:
    root = Path('/home/mmahaut/projects/parrots')
    plots_root = root / 'plots'
    targets = [
        ('Pythia-70m', plots_root / 'ablation_pythia70m' / 'combined_ablation_effects.csv'),
        ('Pythia-1.4b', plots_root / 'ablation_pythia14b' / 'combined_ablation_effects.csv'),
        ('OLMo-1B', plots_root / 'ablation_olmo1b' / 'combined_ablation_effects.csv'),
    ]
    metrics = ['delta_repetition_rate', 'delta_avg_cycle_count']

    for model_name, csv_path in targets:
        df = pd.read_csv(csv_path)
        checkpoints = sorted(
            df['checkpoint'].dropna().astype(str).unique().tolist(),
            key=checkpoint_sort_key,
        )
        for metric in metrics:
            for ck in checkpoints:
                png = csv_path.parent / (
                    f"shared_vs_specific_scatter_{metric_suffix(metric)}_"
                    f"{checkpoint_slug(ck)}_jitter.png"
                )
                friendly_scatter(
                    csv_path,
                    png,
                    model_name,
                    metric,
                    jitter=0.002,
                    checkpoint=ck,
                    show_legends=True,
                )
                print(f'SAVED {png}')

            latest_png = csv_path.parent / (
                f"shared_vs_specific_scatter_{metric_suffix(metric)}_jitter.png"
            )
            friendly_scatter(
                csv_path,
                latest_png,
                model_name,
                metric,
                jitter=0.002,
                checkpoint=checkpoints[-1],
                show_legends=True,
            )
            print(f'SAVED {latest_png}')

    model_data = {m: summarize(pd.read_csv(p)) for m, p in targets}
    panel = plots_root / 'checkpoint_comparison_panel.png'
    abs_panel = plots_root / 'checkpoint_abs_rep_panel.png'
    plot_panel_with_ci(model_data, panel)
    plot_abs_rep_panel_with_ci(model_data, abs_panel)
    print(f'SAVED {panel}')
    print(f'SAVED {abs_panel}')


if __name__ == '__main__':
    main()
