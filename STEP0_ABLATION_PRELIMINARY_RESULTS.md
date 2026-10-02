# Preliminary Step0 (Pre-Training) Head-Ablation Results — Pythia-1.4B

**Status: PRELIMINARY / IN PROGRESS.** This covers only the `step0` checkpoint (the
model's saved random initialization, before any optimizer step) — 96/576 tasks of
the full pythia-1.4b sweep (`step0, step1, step1000, step36000, step71000,
step143000`). The remaining checkpoints are still running as of 2026-09-17. This
document will be updated as later checkpoints complete; do not cite the
per-head/per-layer numbers below as final.

## 1. Setup

- **Model**: `EleutherAI/pythia-1.4b`, HF revision `step0` (the actual saved
  initialization commit for this model, not a synthetic re-initialization).
- **Task**: greedy generation on Minipile-derived prompts, classified into
  **Natural** (prompts whose own greedy continuation degenerates into a
  repeating cycle) and **ICL** (prompts whose greedy continuation does not
  cycle) — classification is checkpoint-specific, i.e. done fresh against
  step0's own generations.
- **Ablation**: one attention head zeroed at a time (pre-output-projection,
  so the intervention isolates a true per-head contribution — see the
  hook-location fix applied earlier in this work), across all 24 layers × 16
  heads = 384 (layer, head) pairs, natural and ICL conditions, ~1000 prompts
  total per layer (sharded into 4 chunks of ~250).
- **Metric**: repetition rate (fraction of generations that cycle) and
  average cycle count, baseline (no ablation) vs. ablated, per (layer, head).

## 2. Headline finding: step0 is already almost entirely repetitive

| Condition | Baseline repetition rate | Baseline avg. cycle count | n prompts |
|---|---|---|---|
| Natural | **95.2%** | 257.5 | 22,392 (pooled across layer/chunk baselines) |
| ICL | **86.6%** | 119.1 | 1,608 |

This is consistent with the untrained-repetition hypothesis: a randomly
initialized transformer collapses into degenerate repetition under greedy
decoding almost regardless of prompt.

**Independent cross-check.** A separate, independently implemented pipeline
(`compute_cycle_descriptive_stats.py`, single-pass generation, no ablation,
1000-token generations, 10-way sharded) measured pythia-1.4b@step0's
cyclic-generation rate at **94.7% (mean over 10 shards of 100 prompts each;
range 92–97%)** — closely matching the 95.2% natural-condition baseline
above, despite different code, different prompt sampling, and much longer
generations (1000 vs ~100 tokens). This cross-validation gives confidence
the baseline number is not an artifact of either pipeline.

Note the ICL condition's small sample (~17 prompts per layer/chunk on
average, 1,608 pooled) reflects how few step0 raw prompts fail to cycle
under this untrained model's own greedy decoding — treat the ICL row as
higher-variance than Natural.

## 3. Per-head ablation effects are weak and diffuse at step0

Across all 384 (layer, head) pairs (Natural condition), single-head
ablation shifts the repetition rate by:

| | Δ repetition rate |
|---|---|
| Mean | −0.0067 (−0.67 pp) |
| Std | 0.0065 |
| Min (largest reduction) | −0.0247 (−2.47 pp) |
| Max (largest increase) | +0.0107 (+1.07 pp) |

**Top 10 heads by repetition reduction (Natural):**

| Layer | Head | Δ repetition rate | Δ avg. cycle count | Ablated rate |
|---|---|---|---|---|
| 23 | 6 | −0.0247 | −2.7 | 0.927 |
| 12 | 7 | −0.0236 | −11.5 | 0.928 |
| 0 | 8 | −0.0236 | −57.2 | 0.928 |
| 0 | 9 | −0.0236 | −42.4 | 0.928 |
| 4 | 5 | −0.0225 | −14.3 | 0.929 |
| 9 | 11 | −0.0204 | −6.7 | 0.931 |
| 5 | 3 | −0.0204 | −6.3 | 0.931 |
| 11 | 0 | −0.0204 | −3.9 | 0.931 |
| 2 | 0 | −0.0204 | −20.8 | 0.931 |
| 8 | 4 | −0.0204 | −15.4 | 0.931 |

**Top 10 heads by repetition increase (Natural):**

| Layer | Head | Δ repetition rate | Δ avg. cycle count | Ablated rate |
|---|---|---|---|---|
| 11 | 13 | +0.0107 | −1.6 | 0.962 |
| 18 | 14 | +0.0107 | +1.0 | 0.962 |
| 12 | 4 | +0.0086 | −1.6 | 0.960 |
| 23 | 15 | +0.0075 | +1.7 | 0.959 |
| 18 | 3 | +0.0064 | +0.2 | 0.958 |
| 12 | 6 | +0.0064 | +5.5 | 0.958 |
| 1 | 15 | +0.0054 | −18.8 | 0.957 |
| 4 | 7 | +0.0054 | −7.5 | 0.957 |
| 4 | 6 | +0.0054 | +3.7 | 0.957 |
| 4 | 14 | +0.0054 | −3.8 | 0.957 |

**Per-layer effects** (mean Δ repetition rate across that layer's 16 heads)
range narrowly from **−0.0087 (layer 13)** to **−0.0038 (layer 0)** — no
layer stands out as disproportionately responsible.

### Interpretation (preliminary)

At step0, no single head or layer is a strong causal driver of repetition:
the largest single-head effect is a ~2.5 percentage-point drop from a ~95%
baseline. This is consistent with the theoretical picture in
`untrained_repetition/README.md` — an untrained transformer's collapse into
repetition looks like a **broad, architecture-wide dynamical property**
(rank collapse / attractor dynamics across many heads and layers), not a
**localized circuit** the way a specific "induction head" might be in a
trained model. Whether this changes at later checkpoints (where a more
concentrated, specialized repetition-related circuit might emerge or
disappear) is exactly what the rest of this sweep (`step1` through
`steplatest`) is designed to test — that comparison is the interesting
result, not step0 in isolation.

## 4. Caveats

- Single run per (layer, head, condition) — no repeated seeds, so per-head
  numbers above have sampling noise; treat individual head rankings as
  suggestive, not significant, until multi-seed or bootstrap CIs are added.
- ICL condition has low n at step0 (see §2) — an untrained model rarely
  produces a non-cyclic "ICL" continuation to begin with.
- This document reflects `step0` only. The full picture (how these effects
  evolve across training) requires the remaining 5 checkpoints, still in
  progress.

*Generated 2026-09-17 from `outputs_ablation_head_cycle_1k_pile/pythia14b/step0/` (96/96 step0 tasks complete) and the `compute_cycle_descriptive_stats.py` step0 cross-check (job 2361859).*
