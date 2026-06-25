# Parrots: Paper Experiment Runbook

This repository contains the code used to analyze how repetition behavior emerges during training in Pythia models. This corresponds to experiments from the paper Repetitions are not all alike: distinct mechanisms sustain repetition in
language models, available on arxiv: https://arxiv.org/abs/2504.01100

## What This README Gives You

1. A map of experiments.
2. For each experiment: how to run data generation and how to produce plots.
3. Expected output files for quick verification.

## Environment Setup

```bash
poetry install
```

## Paper Experiments Index

| ID | Experiment | Run (Data Generation) | Plotting | Main Outputs |
|---|---|---|---|---|
| E1 | Slot-filling evaluation baseline | `python -m parrots.slot_filling data/human_lama_parrots_list_v1.csv EleutherAI/pythia-1.4b outputs/pythia14b_sf` | Integrated in downstream scripts (E2-E6) | `outputs/pythia14b_sf/slot_filling_results.csv` |
| E2 | Dual alluvial (Natural vs No-Cycle-ICL) | Generate checkpoint outputs with `python run_pile_top_p_natural_icl.py --model_name EleutherAI/pythia-1.4b --revision steplatest --output_dir outputs_multihead_full_new --n_samples 256 --max_length 32 --rank 0 --n_ranks 1` (repeat for checkpoints/ranks as needed) | `python run_alluvial_dual.py` | `outputs_multihead_full_new/alluvial_pythia-1.4b_layer_23_dual_alluvial.png` and `.pdf` |
| E3 | Cycle evolution over checkpoints | Use multi-checkpoint outputs under `outputs_multihead_full/` (from the attention pipeline) | `python run_cycle_evolution.py` | `cycle_evolution_horizontal.png` |
| E4 | MLP evolution over checkpoints | Prepare MLP pipeline outputs in `test_mlp_pipeline_output/` | `python run_mlp_evolution.py` | `mlp_evolution_no_step7000.png` |
| E5 | Original vs learnt repetition evolution | `python cycle_evolution_original_vs_learnt.py --model_name EleutherAI/pythia-70m --output_dir cycle_evolution_results` | Integrated in same script; optional alluvial: `python create_repetition_alluvial.py --results_file cycle_evolution_results/cycle_evolution_status_EleutherAI_pythia-70m.json --model_name EleutherAI/pythia-70m --output_dir alluvial_plots` | `cycle_evolution_results/cycle_evolution_original_vs_learnt_EleutherAI_pythia-70m.png`, `alluvial_plots/repetition_alluvial_EleutherAI_pythia-70m.png` |
| E6 | Entropy evolution across checkpoints | `python entropy_evolution_analysis.py --model_name EleutherAI/pythia-70m --output_dir entropy_evolution_results` | Integrated in same script | `entropy_evolution_results/entropy_evolution_EleutherAI_pythia-70m.json`, `.png` |
| E7 | Original-vs-acquired figure panel | Requires E5 JSON files for both `pythia-70m` and `pythia-1.4b` in `cycle_evolution_results/` | `python plot_original_vs_acquired_evolution.py --results-dir cycle_evolution_results --output-dir outputs_multihead_full` | `outputs_multihead_full/original_vs_acquired_repetition_evolution.png` and `.pdf` |

Use the same commands inside `srun` by replacing `python ...` with the wrapper command.

## Recommended Reproduction Order

1. Run E1 if you need fresh slot-filling baselines.
2. Run E2-E4 for the main paper visual tracks (alluvial, cycle evolution, MLP evolution).
3. Run E5-E7 for the original-vs-learnt/acquired decomposition figures.
4. Run E6 for entropy evolution companion analysis.

## Quick Verify Checklist

After each experiment, check that at least one expected output file exists:

1. E2: `*_dual_alluvial.png`
2. E3: `cycle_evolution_horizontal.png`
3. E4: `mlp_evolution_no_step7000.png`
4. E5: `cycle_evolution_original_vs_learnt_*.png`
5. E6: `entropy_evolution_*.png`
6. E7: `original_vs_acquired_repetition_evolution.png`

## File Guide (Paper-Critical)

- Core package: [parrots](parrots)
- Datasets: [data/human_lama_parrots_list_v1.csv](data/human_lama_parrots_list_v1.csv), [data/lama.csv](data/lama.csv)
- Main run scripts: [run_alluvial_dual.py](run_alluvial_dual.py), [run_cycle_evolution.py](run_cycle_evolution.py), [run_mlp_evolution.py](run_mlp_evolution.py)
- Evolution analyses: [cycle_evolution_original_vs_learnt.py](cycle_evolution_original_vs_learnt.py), [entropy_evolution_analysis.py](entropy_evolution_analysis.py)
- Plotting helpers: [create_repetition_alluvial.py](create_repetition_alluvial.py), [plot_original_vs_acquired_evolution.py](plot_original_vs_acquired_evolution.py)
- Extended docs: [docs/01-EXPERIMENTS-OVERVIEW.md](docs/01-EXPERIMENTS-OVERVIEW.md)

## Notes

- Several scripts currently use absolute paths under `/home/mmahaut/projects/parrots`. If you move the repo, update those path constants first.
- The repository includes many generated output directories from prior runs. `.gitignore` is configured to avoid committing heavy artifacts going forward.
