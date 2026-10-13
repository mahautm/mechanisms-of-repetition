# Repetition mechanisms: clean analysis branch

Only the experiments we keep and the code needed for the next steps. Everything else is on
`fix-head-ablation-hook-chaotic`.

## Kept results (data lives in the main checkout; symlinked here, git-ignored)

| Result | Code | Data |
|---|---|---|
| Paired, length-matched generation ablation, Pythia-1.4B, fp16, corrected ablation | `ablation_head_cycle_icl_natural.py --prompt_mode paired`, `slurm/generation_ablation_worker.sh` | `outputs_ablation_paired_fp16/` |
| Paper-design (selected prompts) rerun, fp16, corrected ablation | same, `--prompt_mode selected` | `outputs_ablation_selected_fp16/` |
| Per-head McNemar tests, BH, condition contrasts | `analyze_per_prompt_mcnemar.py` | `*/per_head_mcnemar_all.csv` |
| Initialisation survey (step0 / random init repetition) | `slurm/init_survey_array.sh`, `summarize_init_survey.py` | `outputs_init_survey/` |
| Table 1 (repetition statistics across models) | `compute_cycle_descriptive_stats.py` | `plots/cycle_detection_*_rank*.csv` |
| Figures (paper-style S/D scatter, Δ_ICL vs Δ_Natural, baseline ceiling, Table 1 distributions) | `combine_checkpoint_scatters.py`, `plot_paired_checkpoint_scatter.py`, `plot_proposed_figures.py` | `plots/` |

Head ablation zeroes the head's slice of the *input* of the attention output projection
(before heads are mixed). Results produced before 2026-09-16 used a buggy post-projection hook
and are not kept.

## Next step: teacher-forced ablation (`teacher_forced_ablation.py`)

The generation outcome ("does it repeat within 1,000 tokens") sits at a 92–97% ceiling, so
single-head effects are undetectable. On the same paired prompts we instead fix a greedy
continuation and measure, per position, the change in log-probability of the actual next token,
in logit margin, and in top-1 flips, under:

1. single-head ablation of all heads (zero or mean),
2. per-head attention scores (induction, duplicate-token, previous-token, first-token sink),
3. group ablation of the top-k heads by score vs size-matched random sets.

Regions: `prompt2` (copied half of the ICL prompt / matched Natural text), `pre` (before the
cycle), `cycle` (second cycle onward), `cont` (all generated tokens).

```bash
scripts/submit_teacher_forced.sh                        # Pythia-1.4B, 6 checkpoints, zero-ablation
ABLATION=mean scripts/submit_teacher_forced.sh          # mean-ablation (re-runs baselines)
srun --partition=alien --qos=alien --exclude=node044 \
  bash -lc 'source ~/.bashrc && conda activate parr && python analyze_teacher_forced.py'
```

Primary pre-stated test: across heads, the Spearman correlation between induction score and the
ICL − Natural difference in Δ log p in the `cycle` and `prompt2` regions (permutation p). All
per-head tests are exploratory and BH-corrected.
