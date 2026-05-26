# Causal Audit of Current Results and Plan to Convert Circuit Analysis from Probing to Causal Inference

Date: 2026-03-23

## 1) Executive Summary

The core paper-facing results in this repository are strong descriptive and mechanistic **proxies** (cycle rates, alluvial transitions, head/layer contrasts, lens-based analyses), but they are mostly **non-causal** by strict intervention standards.

The repository also contains substantial intervention work in `cycle-attention-analysis/` (attention manipulation, activation patching, newline removal/addition, semantic-trigger prompting), which is a strong start toward causality. However, most of those studies currently remain **underpowered and claim-heavy** relative to what reviewers usually expect for causal claims in mechanistic interpretability.

Main conclusion: reviewer criticism (“no causal results”) is valid for the paper core, but the codebase already has enough infrastructure to upgrade the circuit section into a credible causal package with moderate refactoring and stronger experimental design.

---

## 2) Scope and Evidence Base

This audit covers:

- Main paper/result pipeline and docs:
  - `README.md`
  - `docs/01-EXPERIMENTS-OVERVIEW.md`
  - `run_alluvial_dual.py`
  - `run_cycle_evolution.py`
  - `cycle_evolution_original_vs_learnt.py`
  - `plot_original_vs_acquired_evolution.py`
  - `multihead_original_vs_acquired.py`
  - `parrots/aa_fortu/aa_fortu.py`
  - `parrots/aa_fortu/aa_fortu_train_multihead_lens.py`

- Causal/intervention branch:
  - `cycle-attention-analysis/README.md`
  - `cycle-attention-analysis/experiments/README.md`
  - `cycle-attention-analysis/experiments/phase1_aggressive_interventions/README.md`
  - `cycle-attention-analysis/experiments/phase4_alternative_mechanisms/README.md`
  - `cycle-attention-analysis/experiments/phase6_final_validation/README.md`
  - `cycle-attention-analysis/reports/phase3_newline_results_analysis.md`
  - `cycle-attention-analysis/data/causal_intervention_L10_H0/causal_intervention_report_L10_H0.md`
  - `cycle-attention-analysis/data/causal_intervention_L15_H0/causal_intervention_report_L15_H0.md`
  - `cycle-attention-analysis/data/activation_patching_L17_19/activation_patching_report_L17_19.md`
  - `cycle-attention-analysis/experiments/early_interventions/causal_attention_intervention.py`
  - `cycle-attention-analysis/experiments/early_interventions/activation_patching_intervention.py`
  - `cycle-attention-analysis/experiments/early_interventions/multi_head_causal_intervention.py`
  - `cycle-attention-analysis/experiments/phase4_alternative_mechanisms/quick_mechanism_explorer.py`
  - `cycle-attention-analysis/experiments/phase4_alternative_mechanisms/phase4_alternative_mechanisms.py`
  - `cycle-attention-analysis/experiments/phase5_semantic_exploitation/phase5_semantic_exploitation.py`
  - `cycle-attention-analysis/experiments/phase6_final_validation/final_validation_test.py`

---

## 3) Causal Grading Rubric Used

I grade each result family on a 0–4 scale:

- **Level 0: Descriptive** (correlation, trend, visualization)
- **Level 1: Predictive/probing** (feature probes/lenses, decoding analyses)
- **Level 2: Interventional but weak** (causal manipulation attempted; limited controls/power)
- **Level 3: Interventional with robust design** (clear controls, adequate N, uncertainty estimates, multiple seeds)
- **Level 4: Mechanistic causal identification** (necessity + sufficiency + mediation/path evidence + robustness)

---

## 4) Audit of Existing Results

## 4.1 Paper-core analyses (main repo)

### A) Repetition evolution and alluvial plots
- Evidence: `run_alluvial_dual.py`, `run_cycle_evolution.py`, `cycle_evolution_original_vs_learnt.py`, `plot_original_vs_acquired_evolution.py`, `OLMO_RESULTS.md`.
- What it gives: temporal patterning of repetition across checkpoints and conditions.
- Causal grade: **Level 0**.
- Why: no intervention on model internals or inputs with counterfactual controls; this is trend analysis.

### B) Attention head/layer “circuit” via contrast heatmaps
- Evidence: `parrots/aa_fortu/aa_fortu.py`, `multihead_original_vs_acquired.py`.
- What it gives: candidate head-level associations with expected-next-token behavior during cycles.
- Causal grade: **Level 1**.
- Why: head contrast/probe scores identify correlates/predictors, not causal necessity/sufficiency.

### C) Lens training and interpretation
- Evidence: `parrots/aa_fortu/aa_fortu_train_multihead_lens.py` and related lens scripts.
- What it gives: decodability/readout of info from hidden/head activations.
- Causal grade: **Level 1**.
- Why: linear/MLP probes are representational readouts; they do not establish functional role.

### D) Entropy / supplementary analyses
- Evidence: entropy scripts listed in docs and overview.
- What it gives: generation uncertainty profiles, condition differences.
- Causal grade: **Level 0**.
- Why: observational summaries, no interventions.

## 4.2 Intervention branch (`cycle-attention-analysis`)

### E) Direct attention and activation interventions (Phase 1 / early interventions)
- Evidence: `causal_attention_intervention.py`, `multi_head_causal_intervention.py`, `activation_patching_intervention.py`, corresponding reports.
- What it gives: first-pass necessity/sufficiency tests for newline/head-based hypotheses.
- Positive: true manipulation exists.
- Main limitations:
  - very small sample defaults (`n_samples` often 5 or 10);
  - limited checkpoint/model coverage in each run;
  - binary/fragile success criteria in places;
  - no systematic uncertainty intervals in reports.
- Causal grade: **Level 2**.

### F) Newline causality testing (Phase 3)
- Evidence: `phase3_newline_results_analysis.md`, fallback comparison scripts.
- What it gives: meaningful negative result (“newlines likely correlational, not causal”).
- Positive: hypothesis falsification through intervention.
- Limitation: still light on statistical power and formal effect estimates.
- Causal grade: **Level 2**.

### G) Semantic trigger discovery/exploitation (Phase 4–6)
- Evidence: `quick_mechanism_explorer.py`, `phase4_alternative_mechanisms.py`, `phase5_semantic_exploitation.py`, `final_validation_test.py`, associated READMEs.
- What it gives: strong prompt-level intervention effects (input manipulations causing repetition increases).
- Positive: this is genuinely interventional at the input level.
- Main limitations for publication-grade causal claims:
  - curated prompts, potential cherry-picking risk;
  - modest sample counts per category;
  - stochastic decoding without broad seed sweeps;
  - denominator smoothing (`baseline + 0.1`) can inflate “x baseline” framing when baseline ≈ 0;
  - rhetoric in docs (“guaranteed”, “deployment ready”) stronger than statistical backing.
- Causal grade: **Level 2 → low Level 3** depending on run design.

---

## 5) Cross-cutting Methodological Gaps (Why Reviewers Say “Not Causal”)

1. **Causal and correlational sections are split**
   - The paper-facing circuit narrative remains largely probe-based; causal tests live in a parallel research branch.

2. **Insufficient necessity/sufficiency decomposition for circuit claims**
   - Candidate heads are identified, but rarely tested in a full matrix:
     - necessity: ablate candidate and measure drop,
     - sufficiency: patch candidate signal into non-repetitive context and measure gain,
     - mediation: test whether candidate carries the effect path.

3. **Power and uncertainty**
   - Several intervention reports rely on small n and no confidence intervals.

4. **Prompt-set confounding**
   - Semantic trigger effects may partly reflect prompt engineering artifacts unless validated on held-out templates, paraphrases, and lexical controls.

5. **Claim calibration**
   - Some documentation uses very strong wording relative to current statistical rigor.

6. **Metric consistency issues**
   - Different scripts use cycle detection outputs differently (counts vs booleans), weakening comparability.

---

## 6) What You Can Claim Now (Defensible)

- Repetition dynamics change across training checkpoints and conditions (descriptive).
- Specific heads/layers are predictive/associated with repetition behavior (probing).
- Newline-based hypotheses receive weak support and likely are non-causal (negative interventional evidence).
- Certain semantic prompt patterns can robustly induce repetition in tested settings (input-level intervention evidence).

## 7) What You Should Not Claim Yet

- “We identified the circuit causally” (not yet demonstrated end-to-end).
- “Head X/Y are the causal mechanism” without necessity+sufficiency+mediation tests.
- “Guaranteed deployment-level effectiveness” without stronger out-of-sample robustness and uncertainty bounds.

---

## 8) Propositions to Convert the Circuit Section from Probing to Causal

This is the key requested redesign.

## Proposition A: Keep probes only for candidate nomination

Use existing probe/lens outputs (`aa_fortu`, multihead heatmaps) as **screening**, not evidence.

Deliverable:
- Candidate head set per checkpoint/condition (top-k by stable association).

## Proposition B: Add necessity tests (targeted ablations)

For each candidate head (and selected head sets):
- zero ablation or mean ablation at cycle-critical positions,
- compare repetition metrics vs matched control heads.

Primary endpoint:
- change in repetition rate / cycle count relative to no-ablation baseline.

Controls:
- random head controls from same layer,
- random position controls,
- non-repetition prompt controls.

## Proposition C: Add sufficiency tests (activation/path patching)

Patch candidate activations from repetitive runs into non-repetitive runs at aligned positions.

Use existing infrastructure from:
- `activation_patching_intervention.py`
- `causal_attention_intervention.py`

but upgrade with:
- larger sample sizes,
- seed sweeps,
- checkpoint sweeps,
- standardized cycle metrics.

## Proposition D: Add mediation/path analysis (“causal circuit” proper)

Test whether effect of semantic triggers on repetition is mediated by candidate heads.

Minimal mediation protocol:
1. Trigger intervention (semantic prompt vs control) increases repetition.
2. Candidate-head ablation reduces that increase.
3. Restoring candidate signal rescues part of the effect.

If steps 1–3 hold robustly, you have mechanistic mediation evidence.

## Proposition E: Statistical rigor package

For each effect report:
- report means + 95% CI,
- include bootstrap/permutation p-values,
- predefine success thresholds,
- include full seed list and prompt list,
- separate discovery vs held-out validation prompts.

## Proposition F: Rewrite circuit claims in paper

Replace “head importance from probes” with:
- “candidate nomination via probes,” and
- “causal validation via intervention matrix (necessity/sufficiency/mediation).”

---

## 9) Concrete Implementation Plan in This Repo

Use existing codebase structure; avoid a full rewrite.

### Phase 1 (fast, 1–2 weeks): Causal retrofit of current circuit section

1. **Standardize metric layer**
   - central function for cycle metrics (rate, count, length, first-cycle position).
   - consume same metric in all intervention scripts.

2. **Create a unified intervention runner**
   - wraps:
     - head ablation,
     - multi-head ablation,
     - activation patching,
     - semantic-trigger interventions.
   - fixed schema outputs (`jsonl` rows with prompt_id, seed, checkpoint, condition, metrics).

3. **Add robust defaults**
   - N >= 100 prompts per condition,
   - >= 5 seeds,
   - checkpoint panel at least `{step1, step1000, step10000, steplatest}`.

4. **Generate causal summary tables/plots**
   - necessity effect sizes by head,
   - sufficiency induction rates by head,
   - mediation plots for top circuits.

### Phase 2 (paper-quality, 2–4 weeks): Causal circuit narrative

1. Probe-nominated candidate circuits per checkpoint.
2. Necessity/sufficiency/mediation validation for top candidates.
3. Held-out template and paraphrase robustness.
4. Cross-model check (Pythia + OLMo at minimum).

---

## 10) Suggested Minimal File Additions

- `scripts/run_causal_circuit_matrix.py`
  - orchestrates necessity/sufficiency/mediation sweeps.

- `parrots/causal_metrics.py`
  - standardized repetition/cycle metrics.

- `parrots/causal_interventions.py`
  - unified ablate/patch primitives with architecture adapters.

- `analyze_causal_circuit_results.py`
  - CI, permutation tests, summary tables, plots.

- `docs/CAUSAL_CIRCUIT_PROTOCOL.md`
  - preregistered protocol and claim language template.

This keeps your existing probing and visualization pipeline intact, while making causal evidence first-class.

---

## 11) Recommended Claim Language (for rebuttal/paper revision)

Use:
- “We use probes to nominate candidate heads and validate them via interventions.”
- “Causal claims are limited to effects reproduced under ablation/patching with controls and uncertainty estimates.”

Avoid:
- “The probe-identified head is causal” (without intervention matrix).
- “Guaranteed effectiveness” unless explicitly bounded to a tested setup.

---

## 12) Bottom Line

You are not starting from zero on causality: the repository already includes real intervention code and informative negative results. The missing piece is integration and rigor: convert probe outputs into candidate selection, then run a standardized necessity+sufficiency+mediation matrix with adequate power and calibrated claims. That change directly addresses the reviewer’s strongest criticism and turns the circuit section from “probing” into “causal mechanistic evidence.”
