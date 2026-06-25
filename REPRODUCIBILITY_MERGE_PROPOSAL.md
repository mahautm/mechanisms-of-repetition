# Reproducibility Merge Proposal: main → chaotic

**Objective:** Selectively merge files from `chaotic` into `main` to achieve complete paper reproducibility without importing exploratory research or dead-end code paths.

**Current Status:**
- ✅ `main` pushed to origin/main (commit 3874af1)
- 📊 `chaotic` contains 1509 additional files
- ⚠️ 70+ paper-critical scripts missing from `main`

**Decision Framework:**
- ✅ **MERGE (Tier 1):** Core paper results & appendix experiments
- ✅ **MERGE (Tier 2):** Multi-model reproduction scripts
- ⚠️ **SKIP:** Exploratory research, failed experiments, redundant variants
- ❌ **DELETE (if in main):** Deprecated or superseded files

---

## PROPOSED DELETIONS FROM MAIN

Based on the [README.md](README.md) paper-critical file list, the following files in `main` are **NOT** paper-relevant and should be deleted:

### Planning & Intermediate Result Documents (DELETE)
These are planning docs and intermediate result snapshots, not part of paper artifact.

```
CAUSAL_AUDIT.md                    → Causal methodology exploration, not in paper
CAUSAL_METHODS_EXPLANATION.md      → Exploratory methods doc (if != published methods, delete)
HEAD_INTERVENTION_EXPERIMENT.md    → Planning for head intervention experiment
ICL_NATURAL_PLAN.md                → Planning document for ICL/Natural split
INTERMEDIATE_RESULTS_2026-04-02.md → Dated intermediate results snapshot
OLMO_ATTENTION_FALLBACK_PLAN.md   → Planning fallback, not final reproducibility
OLMO_RESULTS.md                    → Intermediate results doc (use chaotic version if needed)
```

**Rationale:** These are working documents, not part of the reproducible paper artifact. Users don't need intermediate planning notes to run experiments.

---

### Top-Level Exploratory Analysis Scripts (DELETE)

These are exploratory analyses not mentioned in [README.md](README.md) paper experiments table.

```
analyze_2x2_natural_icl_original_acquired.py  → Exploratory 2x2 analysis variant
analyze_pile_mitigation_efficiency.py         → Exploratory efficiency analysis
logit_entropy_analysis.py                     → Exploratory entropy analysis (not in paper)
logit_entropy_per_checkpoint.py               → Exploratory variant
multihead_original_vs_acquired.py             → Exploratory multihead analysis
multihead_original_vs_acquired_evolution.py   → Exploratory evolution variant
plot_all_layers_original_vs_acquired.py       → Exploratory plotting variant
plot_checkpoint_comparison_panel.py           → Exploratory panel plotting
plot_innate_vs_acquired_evolution.py          → Exploratory innate-vs-acquired variant
plot_user_friendly_scatter.py                 → User-friendly scatter plot (not in paper)
score_selective_head_policy.py                → Exploratory head policy scoring
```

**Rationale:** README lists only 2 plotting helpers: `create_repetition_alluvial.py` and `plot_original_vs_acquired_evolution.py`. Others are exploratory variants not in final paper.

---

### Old Variant Implementations in parrots/aa_fortu/ (DELETE)

These are old implementations superseded by `multihead_analysis_graphs.py` and training utilities in `chaotic`.

```
parrots/aa_fortu/aa_fortu.py                   → Original/old version (superseded)
parrots/aa_fortu/aa_fortu_distib_graph.py     → Variant (distributed graph)
parrots/aa_fortu/aa_fortu_graphs2.py          → Variant (graphs v2)
parrots/aa_fortu/aa_fortu_train_lens.py       → Old lens training variant
parrots/aa_fortu/aa_fortu_train_lens_2.py     → Old lens training variant v2
parrots/aa_fortu/aa_fortu_train_mlp_lens.py   → Old MLP lens training
parrots/aa_fortu/aa_fortu_train_multihead_lens.py → Old multihead lens training
parrots/aa_fortu/aa_head_per_layers_graph.py  → Old per-layer head graph
parrots/aa_fortu/aa_pairs.py                  → Exploratory pairs analysis
parrots/aa_fortu/attention_analysis.py        → Exploratory attention analysis
parrots/aa_fortu/attention_regroup.py         → Exploratory attention regrouping
parrots/aa_fortu/attention_through_facts.py   → Exploratory facts-based analysis
parrots/aa_fortu/attention_through_pile.py    → Exploratory PILE-based analysis
parrots/aa_fortu/ckpt_pipeline.py             → Old checkpoint pipeline v1
parrots/aa_fortu/ckpt_pipeline_2.py           → Old checkpoint pipeline v2
parrots/aa_fortu/ckpt_pipeline_3.py           → Old checkpoint pipeline v3
parrots/aa_fortu/ckpt_pipeline_id.py          → Old checkpoint pipeline ID variant
parrots/aa_fortu/ckpt_pipeline_id_graph.py    → Old checkpoint pipeline ID graph
parrots/aa_fortu/ckpt_sentence_extractor.py   → Exploratory sentence extraction
```

**Rationale:** All canonical logic is now in `multihead_analysis_graphs.py`. These variants accumulated during development and are not needed for reproducibility.

---

### Exploratory parrots/ Utilities (DELETE)

These are exploratory analysis/metric scripts not used by paper-critical pipelines:

```
parrots/_random_test.py                 → Random testing file (debug only)
parrots/2d_mapping_all.py              → Exploratory 2D mapping
parrots/2d_mapping_layers.py           → Exploratory 2D mapping per layer
parrots/2d_mapping_loop.py             → Exploratory 2D mapping loop
parrots/autoprompt_frequency.py        → Exploratory autoprompt analysis
parrots/av_seq_graph.py                → Exploratory avg sequence graphing
parrots/bins_repetitions_proportion.py → Exploratory binning analysis
parrots/cycle_perturbation.py          → Different perturbation method (exploratory)
parrots/density_of_sequence.py         → Exploratory density analysis
parrots/dii_ranking_tokens.py          → Exploratory token ranking by DII
parrots/distance.py                    → Exploratory distance metrics
parrots/distance_analysis.py           → Exploratory distance analysis
parrots/distance_autop.py              → Exploratory autoprompt distance
parrots/distance_heatmap.py            → Exploratory distance heatmap
parrots/extract_all_hidden.py          → Exploratory hidden state extraction
parrots/extract_average_hidden.py      → Exploratory averaged hidden extraction
parrots/graph.py                       → Exploratory graph utilities
parrots/long_term_memory.py            → Exploratory LTM analysis
parrots/nli.py                         → Exploratory NLI analysis
parrots/pile_perturbations.py          → Exploratory PILE-based perturbations
parrots/r_dist_sequence_rand.py        → Exploratory random dist analysis
parrots/rank_distance_sequence.py      → Exploratory rank distance
parrots/scores_cleanup.py              → Data cleanup utility (not pipeline)
parrots/sf_transition_test.py          → Exploratory SF transition test
parrots/top_5_ranking_tokens.py        → Exploratory token ranking
parrots/training_data_confidence.py    → Exploratory confidence analysis
```

**Rationale:** These are scattered analytical explorations. None appear in README paper experiments or are called by main pipeline scripts.

---

### Entire Exploratory Research Directory (DELETE)

```
parrots/rr/                           → Entire directory of random repetition research
  random_r_aa_graph.py
  random_rep.py
  random_rep_attention_analysis.py
  random_rep_graph.py
  repetition_analysis.py
  rraa_coactivation.py
  rraa_first_loop.py
  rraa_first_loop_human.py
  rraa_graph_heatmap.py
  rraa_graph_multi_idx.py
  rraa_neurone_progression.py
  test_rraa_graph.py
```

**Rationale:** Entire research track investigating random repetition. Not mentioned in paper. Can be archived in separate branch.

---

### Utility & Variant Shell Scripts in scripts/ (DELETE)

Most `.sh` files are old job submission scripts or exploratory. Keep only critical ones:

**DELETE these:**
```
scripts/aa_fortu.sh                              → Old aa_fortu job variant
scripts/aa_fortu_ckpts.sh                        → Old checkpoint job variant
scripts/aa_fortu_ckpts_ID.sh                     → Old ID-based ckpt variant
scripts/aa_fortu_ckpts_genLen.sh                 → Old gen-len variant
scripts/aa_fortu_train.sh                        → Old training job
scripts/aa_fortu_train_ckpts.sh                  → Old training checkpoint job
scripts/at_pile.sh                              → Old PILE analysis job
scripts/diagnose_alluvial_table_mismatch.py     → Diagnostic utility (not paper)
scripts/hidden_extractor.sh                     → Old hidden extraction
scripts/pert_analy_distrib.sh                    → Old perturbation analysis job
scripts/perturbated.sh                          → Typo'd old job
scripts/pre_dist.sh                             → Old preprocessing job
scripts/quick_icl_natural_test.sh               → Quick test (not reproducibility)
scripts/reps.sh                                 → Old reps job
scripts/reps_aa.sh                              → Old aa-reps job
scripts/reps_aa_pyth.sh                         → Old aa-pythia job
scripts/reps_pyth.sh                            → Old pythia reps job
scripts/reps_pyth_olm.sh                        → Old OLMo reps job
scripts/run_2x2_analysis.sh                     → Old 2x2 analysis
scripts/run_ablation_extensive_matrix.sh        → Old extensive ablation
scripts/run_ablation_extensive_matrix_parallel.sh → Old parallel variant
scripts/run_ablation_head_cycle_full.sh         → Old full ablation (use chaotic)
scripts/run_ablation_head_cycle_smoke.sh        → Old smoke test
scripts/run_ablation_natural_to_no_cycle_icl.sh → Old natural/ICL ablation
scripts/run_ablation_tier2_icl_parallel.sh      → Old tier2 parallel
scripts/run_alluvial_data_pythia70m.sh          → Old Pythia70m job
scripts/run_alluvial_multicategory.sh           → Old multicategory
scripts/run_cyc0_pythia70m.sh                   → Old cycle detection job
scripts/run_entropy_3way_1.4b.sh                → Old entropy job (use chaotic)
scripts/run_entropy_3way_70m.sh                 → Old entropy job (use chaotic)
scripts/run_entropy_evolution.sh                → Old entropy evolution job
scripts/run_entropy_per_checkpoint.sh           → Old entropy per ckpt
scripts/run_mh_evo_all_layers_1.4b.sh          → Old multihead evolution
scripts/run_mh_evo_all_layers_70m.sh           → Old multihead evolution
scripts/run_multihead_graph_generation.sh       → Old graph generation
scripts/run_olmo_icl_natural_rerun_l4_l15.sh    → Old OLMo rerun
scripts/run_on_compute.sh                       → Old compute runner
scripts/run_rank_*.sh (all 8 files)             → Old ranking jobs (not in paper)
scripts/run_rank_cycle_table.sh                 → Old cycle table ranking
scripts/run_rank_cycle_table_1000.sh            → Old cycle table ranking variant
scripts/sf.sh                                   → Old SF job
scripts/sf_causal.sh                            → Old causal SF job
scripts/slurm_sf.sh                             → Old SLURM SF job
scripts/submit_all_512.sh                       → Old 512 submission
scripts/submit_all_models_table.sh              → Old model table submission
scripts/submit_mitigations.sh                   → Old mitigation submission
scripts/test_ablation_icl_natural.sh            → Test file
scripts/test_dimensions.sh                      → Test file
scripts/test_mlp_lens.sh                        → Test file
scripts/test_mlp_pipeline.sh                    → Test file
scripts/train_mlp_lenses.sh                     → Old training job (get from chaotic)
scripts/train_multihead_lenses.sh               → Old training job (get from chaotic)
scripts/train_multihead_lenses_pythia70m.py    → Old training job (get from chaotic)
scripts/transitions.sh                          → Old transitions job
```

**KEEP these:**
```
scripts/aggregate_cycles.py           → Paper Table 1 (will be in chaotic merge)
scripts/write_table_from_aggregated.py → Table formatting (will be in chaotic merge)
scripts/compare_pythia_alluvial.py    → Alluvial comparison (will be in chaotic merge)
scripts/compare_alluvial_table.py     → Table comparison (will be in chaotic merge)
scripts/run_full_multihead_analysis.sh → Core multihead job runner (assess if used)
scripts/run_multihead_analysis.sh     → Core multihead runner (assess if used)
scripts/run_multihead_analysis_pythia70m.sh → Core Pythia70m runner (assess if used)
scripts/run_pythia70m_multihead.sh    → Core Pythia70m multihead (assess if used)
scripts/run_cycle_evolution_orig_learnt.sh → Core cycle evolution (assess if used)
```

**Rationale:** Most are legacy job runners from exploratory phase. Keep only those explicitly called by main plotting scripts or paper reproducibility.

---

### Documentation Files (ASSESS)

```
docs/00-STRUCTURE-OVERVIEW.md        → May be redundant after README update
docs/01-EXPERIMENTS-OVERVIEW.md      → Likely redundant with README
docs/02-USING-OLMO.md                → Keep (OLMo setup), or merge into experiments/
docs/03-QUICK-START.md               → Likely redundant with README
docs/README.md                       → May be redundant
```

**Recommendation:** Consolidate `docs/` into top-level [README.md](README.md) or into `experiments/README.md`. Delete individual docs unless they contain essential setup info not in README.

---

### Slot-Filling Related (ASSESS)

```
parrots/sf_perf.py                   → SF performance analysis (exploratory?)
parrots/sf_from_causal.py            → SF from causal analysis (exploratory?)
```

**Rationale:** Paper does use slot-filling (E1), but these might be variants. Verify if called by main pipeline before deleting.

---

### Prepare/Helper Scripts (ASSESS)

```
parrots/prepare_lama.py              → LAMA dataset preparation (used by E1?)
```

**Rationale:** If used for E1 data prep, keep. If exploratory, delete.

---

## CONSOLIDATED DELETION LIST

| File/Pattern | Type | Action |
|---|---|---|
| `CAUSAL_*.md`, `HEAD_INTERVENTION_*.md`, `ICL_NATURAL_*.md`, `INTERMEDIATE_*.md`, `OLMO_ATTENTION_FALLBACK_*.md`, `OLMO_RESULTS.md` | Doc | ❌ DELETE |
| `analyze_2x2_*.py`, `analyze_pile_*.py`, `logit_*.py`, `multihead_*_evolution.py`, `plot_all_layers_*.py`, `plot_checkpoint_*.py`, `plot_innate_*.py`, `plot_user_*.py`, `score_*.py` | Script | ❌ DELETE |
| `parrots/aa_fortu/aa_*.py`, `parrots/aa_fortu/ckpt_*.py`, `parrots/aa_fortu/attention_*.py` (11+ variants) | Old impl | ❌ DELETE |
| `parrots/2d_mapping_*.py`, `parrots/autoprompt_*.py`, `parrots/av_seq_*.py`, `parrots/bins_*.py`, `parrots/cycle_perturbation.py`, `parrots/density_*.py`, ... (25+ exploratory) | Explor | ❌ DELETE |
| `parrots/rr/*` (12 files) | Research | ❌ DELETE |
| `scripts/*.sh` (50+ old job runners) | Old job | ❌ DELETE |
| `docs/*.md` (5 files) | Doc | ⚠️ CONSOLIDATE |
| `parrots/sf_perf.py`, `parrots/sf_from_causal.py` | Variant | ⚠️ ASSESS |

**Total files to delete: ~130 files**
**Total files to consolidate: ~5 files**
**Total files to assess: ~2 files**

---

## DELETION STRATEGY

### Phase 1: Safe Deletions (High Confidence)
Delete files with high confidence they're exploratory:
- All `CAUSAL_*.md`, `HEAD_INTERVENTION_*.md`, `ICL_NATURAL_*.md`, etc.
- All `analyze_2x2_*.py`, `logit_*.py`, `multihead_*_evolution.py`
- All `parrots/2d_mapping_*.py`, `parrots/autoprompt_*.py`, etc.
- Entire `parrots/rr/` directory
- 50+ old shell scripts in `scripts/`

### Phase 2: Variant Review (Medium Confidence)
Before deleting aa_fortu/ variants and old pipeline implementations:
- Verify `multihead_analysis_graphs.py` is canonical
- Confirm no other scripts import from old variants
- Check git history for which were last modified

### Phase 3: Assessment (Low Confidence)
- Verify `parrots/prepare_lama.py` is used by E1
- Verify `parrots/sf_perf.py`, `sf_from_causal.py` aren't called by paper plots
- Consolidate `docs/*.md` into README or experiments guide

---

## PROPOSED ADDITIONS FROM CHAOTIC

### TIER 1: CRITICAL FOR PAPER REPRODUCIBILITY

#### 1.1 Cycle Statistics & Prevalence Table
```
scripts/aggregate_cycles.py           → Generates cycle stats per model/checkpoint
scripts/write_table_from_aggregated.py → Formats LaTeX table output
scripts/compare_pythia_alluvial.py    → Cross-model cycle comparison
scripts/compare_alluvial_table.py     → Alluvial stat aggregation
```
**Rationale:** Paper Table 1 "Cycle Prevalence Across Training" depends on these. Currently only `run_cycle_evolution.py` exists; these compute aggregated stats.

**Paper Section:** Main text, Table 1

#### 1.2 Checkpoint-Level Descriptive Statistics
```
compute_cycle_descriptive_stats.py    → Per-checkpoint cycle distribution stats
compute_intro_cycle_stats.py          → Introductory checkpoint snapshot stats
calc_apertus_stats.py                 → Apertus-specific stat computation
```
**Rationale:** Support aggregation scripts for prevalence table. Provide granular breakdowns.

**Paper Section:** Main text, Table 1 & Appendix

#### 1.3 Alluvial Development Trajectory Figures
```
create_dual_alluvial_from_cycle_evolution.py → Generates Figure 2 (main paper figure)
```
**Rationale:** Creates dual natural/ICL alluvial plots. Core visualization of repetition evolution.

**Paper Section:** Figure 2 (main text)

#### 1.4 Attention Head Ablation Analysis
```
ablation_head_cycle_evolution.py      → Runs head ablation jobs (Natural condition)
ablation_head_cycle_icl_natural.py    → Runs head ablation jobs (ICL condition)
combine_checkpoint_scatters.py        → Aggregates ablation results across checkpoints
combine_cycle_grid_results.py         → Cross-cycle ablation analysis
build_combined_ablation_generic.py    → Generic multi-model ablation aggregation
build_apertus_combined_ablation.py    → Apertus-specific ablation rollup
analyze_ablation_head_cycle.py        → Delta computation & significance testing
```
**Rationale:** README mentions ablation but code doesn't execute it end-to-end. These scripts complete the ablation pipeline.

**Paper Section:** Figure 3 (attention head role), Appendix ablation results

#### 1.5 Head-Policy Mitigation (Appendix)
```
prepare_head_policy_library.py        → Builds intervention policy library
plot_head_policy_matrix_results.py    → Generates head-policy ablation matrix plot
```
**Rationale:** Appendix experiments on top-p mitigation via head intervention. Completes perturbation story.

**Paper Section:** Appendix - Head Policy Ablation

#### 1.6 Entropy & Confidence Analysis
```
entropy_analysis.py                   → Core entropy computation
entropy_3way_categorization.py        → 3-way categorical entropy breakdown
replot_entropy_no_icl.py             → Replots entropy without ICL condition
```
**Rationale:** Confidence & entropy section in paper. `entropy_evolution_analysis.py` exists but these provide analysis tools.

**Paper Section:** Main text - Confidence & Entropy

---

### TIER 2: MULTI-MODEL REPRODUCTION & APPENDIX

#### 2.1 OLMo Attention Analysis Pipeline
```
experiments/olmo_attention_pipeline.py              → OLMo reproducibility wrapper
experiments/run_olmo_attention_analysis.sh          → OLMo batch script
experiments/run_olmo_checkpoints.sh                 → OLMo checkpoint iteration
```
**Rationale:** Paper includes OLMo results. Users need end-to-end pipeline to reproduce.

**Paper Section:** Appendix - Multi-model Results (OLMo)

#### 2.2 Apertus Attention Analysis Pipeline
```
experiments/run_apertus_checkpoints.sh             → Apertus checkpoint iteration
launch_apertus_ablation_all_layers.sh              → Apertus ablation batch job
run_apertus_ablation_all_layers_array.sh           → Array worker variant
run_Apertus_512.sh, run_Apertus_icl.sh, etc.      → Per-model Apertus runners
```
**Rationale:** Paper includes Apertus results. These enable reproduction.

**Paper Section:** Appendix - Multi-model Results (Apertus)

#### 2.3 OLMo Plot Variants
```
plot_olmo_alluvial_dual.py                         → Dual alluvial for OLMo
plot_olmo_alluvial.py                              → Single alluvial for OLMo
plot_olmo_instruct_alluvial_dual.py               → OLMo-Instruct variant
plot_apertus_alluvial_dual.py                      → Dual alluvial for Apertus
```
**Rationale:** Multi-model visualization. Enables appendix figure reproduction.

**Paper Section:** Appendix - Multi-model Figures

#### 2.4 Per-Model Batch Runners
```
run_OLMo_512.sh, run_OLMo_icl.sh, run_OLMo_p0.5.sh, run_OLMo_p0.9.sh
run_Apertus_512.sh, run_Apertus_icl.sh, run_Apertus_p0.5.sh, run_Apertus_p0.9.sh
run_Llama_512.sh, run_Llama_icl.sh, run_Llama_p0.5.sh, run_Llama_p0.9.sh
run_Pythia-1.4B_icl.sh, run_Pythia-1.4B_p0.5.sh, run_Pythia-1.4B_p0.9.sh
```
**Rationale:** Enable reproduction for Llama, Pythia-1.4B, and other model variants mentioned in paper.

**Paper Section:** Appendix - Multi-model Experiments

### TIER 3: UTILITIES & WORKFLOW SUPPORT (LOWER PRIORITY)

#### 3.1 Cycle Detection Extension
```
run_cycle_detection_for_new_models.py              → Generic model cycle detection
```
**Rationale:** Extensibility. Enables users to apply method to new models.

**Paper Section:** Appendix - Future Work / Reproducibility

#### 3.2 Analysis & Plotting Utilities
```
format_latex.py                                   → LaTeX table formatting
log_analyzer.py                                   → Batch job log analysis
replot_non_apertus.py                             → Replot fallback for non-Apertus
merge_batched_outputs.py                          → Aggregates scattered output files
merge_ablation_layers_generic.py                  → Generic layer ablation merging
merge_apertus_ablation_layers.py                  → Apertus-specific layer merging
```
**Rationale:** Workflow utilities to regenerate figures from raw outputs. Lower priority than experiment drivers.

**Paper Section:** Internal Tools

#### 3.3 Debugging & Sanity Checks
```
test_generate_and_detect.py                       → Generation + cycle detection sanity check
test_eager.py, test_eager_no.py                   → PyTorch eager mode tests
debug_apertus_generation.py                       → Apertus generation debugging
sanity_apertus_long_greedy.py                     → Sanity check for greedy sampling
```
**Rationale:** Diagnostic tools. Include for reproducibility validation but not blocking.

**Paper Section:** Appendix - Validation

#### 3.4 Miscellaneous & Exploratory
```
patch.py                                          → One-off patch utility (assess case-by-case)
generate_ablation_conclusions.py                  → Ablation summary generator
rerun_missing_apertus_jobs.py                     → Job recovery utility
sbatch_apertus_v2.sh                              → Alternative SLURM variant
```
**Rationale:** Case-by-case inclusion. Likely safe to include but verify they don't conflict.

---

## PROPOSED DELETIONS FROM CHAOTIC (DON'T MERGE)

### Exploratory Research (Not in Paper)
```
cycle-attention-analysis/                         → Separate research branch (100+ files, many binary)
                                                    Rationale: Not mentioned in paper. Archive separately.

untrained_repetition/                            → Untrained model research (separate track)
                                                    Rationale: Separate from main paper narrative.
                                                    Decision: Document but don't merge to main.
```

### Redundant or Deprecated
```
run_mlp_evolution.sh                              → Superseded by run_mlp_evolution.py (already in main)
aggregate_ablation_pythia70m.py                   → Empty file (0 bytes)
analyze_ablation_icl_natural.py                   → Empty file (0 bytes)
```

### Large Script Collections (Use Selectively)
```
launch_ablation_all_models_1k.sh                  → Subset of launch scripts; use only if needed
launch_compute_cycles_interactive.sh              → Interactive debugging; skip
launch_*.sh files (general)                       → Use specific ones only, not all
submit_*.sh files (general)                       → Use specific ones only
run_ablation_all_models_1k_array_worker.sh       → Internal worker script; skip if no master
run_head_policy_array_worker.sh                   → Array worker; use only with master job
```

---

## FILE-BY-FILE ACTION MATRIX

| File/Pattern | Action | Priority | Rationale |
|---|---|---|---|
| `scripts/aggregate_cycles.py` | ✅ MERGE | P0 | Table 1 generation |
| `scripts/write_table_from_aggregated.py` | ✅ MERGE | P0 | Table 1 formatting |
| `compute_cycle_descriptive_stats.py` | ✅ MERGE | P0 | Cycle stats |
| `compute_intro_cycle_stats.py` | ✅ MERGE | P1 | Intro stats |
| `create_dual_alluvial_from_cycle_evolution.py` | ✅ MERGE | P0 | Figure 2 |
| `ablation_head_cycle_*.py` | ✅ MERGE | P0 | Head ablation experiments |
| `combine_checkpoint_scatters.py` | ✅ MERGE | P0 | Ablation aggregation |
| `combine_cycle_grid_results.py` | ✅ MERGE | P1 | Cross-cycle analysis |
| `build_*_ablation*.py` | ✅ MERGE | P1 | Multi-model ablation |
| `analyze_ablation_head_cycle.py` | ✅ MERGE | P1 | Ablation delta computation |
| `entropy_analysis.py` | ✅ MERGE | P1 | Entropy section |
| `entropy_3way_categorization.py` | ✅ MERGE | P1 | Categorical entropy |
| `replot_entropy_no_icl.py` | ✅ MERGE | P2 | Entropy figure fallback |
| `experiments/olmo_*.py` | ✅ MERGE | P1 | OLMo reproducibility |
| `experiments/run_olmo_*.sh` | ✅ MERGE | P1 | OLMo batch jobs |
| `experiments/*.md` | ✅ MERGE | P1 | OLMo/Apertus guides |
| `plot_olmo_alluvial*.py` | ✅ MERGE | P1 | OLMo plotting |
| `plot_apertus_alluvial*.py` | ✅ MERGE | P1 | Apertus plotting |
| `run_OLMo_*.sh` | ✅ MERGE | P2 | OLMo per-model runners |
| `run_Apertus_*.sh` | ✅ MERGE | P2 | Apertus per-model runners |
| `run_Llama_*.sh` | ✅ MERGE | P2 | Llama per-model runners |
| `format_latex.py` | ✅ MERGE | P3 | Table formatting utility |
| `log_analyzer.py` | ✅ MERGE | P3 | Log analysis utility |
| `run_cycle_detection_for_new_models.py` | ✅ MERGE | P3 | Extensibility |
| `cycle-attention-analysis/` | ❌ SKIP | — | Separate research branch |
| `untrained_repetition/` | ❌ SKIP | — | Separate research track |
| `aggregate_ablation_pythia70m.py` | ❌ SKIP | — | Empty file (0 bytes) |
| `analyze_ablation_icl_natural.py` | ❌ SKIP | — | Empty file (0 bytes) |
| `launch_ablation_all_models_1k.sh` | ⚠️ ASSESS | — | Use if needed for pipeline |
| `launch_compute_cycles_interactive.sh` | ❌ SKIP | — | Debugging only |

---

## MIGRATION STRATEGY

### Phase 1: Critical Path (P0-P1)
Merge **35-40 files** covering:
- Cycle statistics & table generation
- Head ablation pipeline
- Alluvial figures
- Entropy analysis
- OLMo/Apertus core scripts

**Estimated lines of code:** ~4,000

### Phase 2: Multi-Model Support (P2)
Merge **20-25 per-model runners** for OLMo, Apertus, Llama

**Estimated lines of code:** ~1,000

### Phase 3: Utilities & Debugging (P3)
Merge **10-15 utility scripts** for plotting, logging, extensibility

**Estimated lines of code:** ~800

### Phase 4: Documentation Updates
Update [README.md](README.md) to reference:
- `parrots/aa_fortu/multihead_analysis_graphs.py` as canonical for Figure 2a (already correct)
- New scripts for Table 1, ablation figures, entropy analysis
- OLMo/Apertus reproduction instructions in appendix section

---

## VALIDATION CHECKLIST

After merge, verify:
- [ ] All P0 scripts execute without import errors (run with `poetry run python <script> --help`)
- [ ] Cycle aggregation produces Table 1 output format
- [ ] Alluvial creation generates expected PNG files
- [ ] Head ablation delta computation completes
- [ ] Entropy analysis produces entropy figure
- [ ] OLMo pipeline runs through checkpoint set without failure
- [ ] Apertus pipeline runs through checkpoint set without failure
- [ ] README lists all paper figures with corresponding scripts

---

## EXECUTION PLAN

```bash
# Step 1: Create temporary branch for cleanup + merge
git checkout -b refactor/main-cleanup-and-merge

# Step 2: Delete exploratory/non-paper files from main
# Phase 1: Delete planning docs
git rm CAUSAL_AUDIT.md CAUSAL_METHODS_EXPLANATION.md HEAD_INTERVENTION_EXPERIMENT.md \
         ICL_NATURAL_PLAN.md INTERMEDIATE_RESULTS_2026-04-02.md \
         OLMO_ATTENTION_FALLBACK_PLAN.md OLMO_RESULTS.md

# Phase 2: Delete exploratory analysis scripts
git rm analyze_2x2_natural_icl_original_acquired.py analyze_pile_mitigation_efficiency.py \
         logit_entropy_analysis.py logit_entropy_per_checkpoint.py \
         multihead_original_vs_acquired.py multihead_original_vs_acquired_evolution.py \
         plot_all_layers_original_vs_acquired.py plot_checkpoint_comparison_panel.py \
         plot_innate_vs_acquired_evolution.py plot_user_friendly_scatter.py \
         score_selective_head_policy.py

# Phase 3: Delete old aa_fortu variants
git rm parrots/aa_fortu/aa_fortu.py parrots/aa_fortu/aa_fortu_distib_graph.py \
         parrots/aa_fortu/aa_fortu_graphs2.py parrots/aa_fortu/aa_fortu_train_*.py \
         parrots/aa_fortu/aa_head_per_layers_graph.py parrots/aa_fortu/aa_pairs.py \
         parrots/aa_fortu/attention_analysis.py parrots/aa_fortu/attention_regroup.py \
         parrots/aa_fortu/attention_through_facts.py parrots/aa_fortu/attention_through_pile.py \
         parrots/aa_fortu/ckpt_pipeline*.py parrots/aa_fortu/ckpt_sentence_extractor.py

# Phase 4: Delete exploratory parrots utilities
git rm parrots/_random_test.py parrots/2d_mapping_*.py parrots/autoprompt_frequency.py \
         parrots/av_seq_graph.py parrots/bins_repetitions_proportion.py \
         parrots/cycle_perturbation.py parrots/density_of_sequence.py \
         parrots/dii_ranking_tokens.py parrots/distance*.py \
         parrots/extract_all_hidden.py parrots/extract_average_hidden.py \
         parrots/graph.py parrots/long_term_memory.py parrots/nli.py \
         parrots/pile_perturbations.py parrots/r_dist_sequence_rand.py \
         parrots/rank_distance_sequence.py parrots/scores_cleanup.py \
         parrots/sf_transition_test.py parrots/top_5_ranking_tokens.py \
         parrots/training_data_confidence.py

# Phase 5: Delete entire random-repetition research branch
git rm -r parrots/rr/

# Phase 6: Delete old shell scripts from scripts/ (keep only critical ones)
cd scripts && git rm aa_fortu.sh aa_fortu_ckpts.sh aa_fortu_ckpts_ID.sh \
                   aa_fortu_ckpts_genLen.sh aa_fortu_train.sh aa_fortu_train_ckpts.sh \
                   at_pile.sh diagnose_alluvial_table_mismatch.py hidden_extractor.sh \
                   pert_analy_distrib.sh perturbated.sh pre_dist.sh quick_icl_natural_test.sh \
                   recompute_detect_on_alluvial.py reps*.sh run_2x2_analysis.sh \
                   run_ablation_extensive_matrix*.sh run_ablation_head_cycle_full.sh \
                   run_ablation_head_cycle_smoke.sh run_ablation_natural_to_no_cycle_icl.sh \
                   run_ablation_tier2_icl_parallel.sh run_alluvial_data_pythia70m.sh \
                   run_alluvial_multicategory.sh run_cyc0_pythia70m.sh \
                   run_entropy_3way_*.sh run_entropy_evolution.sh run_entropy_per_checkpoint.sh \
                   run_mh_evo_all_layers_*.sh run_multihead_graph_generation.sh \
                   run_olmo_icl_natural_rerun_l4_l15.sh run_on_compute.sh \
                   run_rank_*.sh sf.sh sf_causal.sh slurm_sf.sh \
                   submit_all_512.sh submit_all_models_table.sh submit_mitigations.sh \
                   test_*.sh transitions.sh && cd ..

# Phase 7: Consolidate/cleanup docs/ (optional: remove if content moved to README)
# git rm docs/00-STRUCTURE-OVERVIEW.md docs/01-EXPERIMENTS-OVERVIEW.md \
#          docs/03-QUICK-START.md docs/README.md

# Step 3: Commit cleanup
git add -A
git commit -m "Clean up exploratory scripts and planning docs

Removes:
- Planning documents (CAUSAL_*.md, HEAD_INTERVENTION_*.md, etc.)
- Exploratory analysis scripts (analyze_2x2_*, logit_*, multihead_*_evolution.py, plot_* variants)
- Old aa_fortu implementations (aa_*, ckpt_pipeline* variants)
- Exploratory parrots utilities (2d_mapping_*, dii_*, distance*, etc.)
- Entire parrots/rr/ research branch
- 50+ legacy shell scripts from scripts/

Keeps: core paper-reproducible scripts (slot_filling, cycle_detection,
multihead_analysis_graphs, perturbation_analysis, entropy_evolution, etc.)

See REPRODUCIBILITY_MERGE_PROPOSAL.md for detailed list."

# Step 4: Cherry-pick P0-P1 files from chaotic (critical reproducibility scripts)
# See TIER 1 section of REPRODUCIBILITY_MERGE_PROPOSAL.md

git cherry-pick chaotic -- scripts/aggregate_cycles.py scripts/write_table_from_aggregated.py \
                          scripts/compare_pythia_alluvial.py scripts/compare_alluvial_table.py \
                          compute_cycle_descriptive_stats.py compute_intro_cycle_stats.py \
                          calc_apertus_stats.py \
                          create_dual_alluvial_from_cycle_evolution.py \
                          ablation_head_cycle_evolution.py ablation_head_cycle_icl_natural.py \
                          combine_checkpoint_scatters.py combine_cycle_grid_results.py \
                          build_combined_ablation_generic.py build_apertus_combined_ablation.py \
                          analyze_ablation_head_cycle.py \
                          entropy_analysis.py entropy_3way_categorization.py replot_entropy_no_icl.py \
                          prepare_head_policy_library.py plot_head_policy_matrix_results.py

# Step 5: Cherry-pick P2 files from chaotic (multi-model support)
git cherry-pick chaotic -- experiments/olmo_attention_pipeline.py \
                          experiments/run_olmo_attention_analysis.sh experiments/run_olmo_checkpoints.sh \
                          experiments/run_apertus_checkpoints.sh launch_apertus_ablation_all_layers.sh \
                          plot_olmo_alluvial_dual.py plot_apertus_alluvial_dual.py \
                          plot_olmo_alluvial.py plot_olmo_instruct_alluvial_dual.py

# Step 6: Test on cluster
poetry run python scripts/aggregate_cycles.py --help
poetry run python create_dual_alluvial_from_cycle_evolution.py --help
poetry run python entropy_analysis.py --help

# Step 7: Update README with new scripts & canonical paths
# See "README UPDATES NEEDED" section below

# Step 8: Push to origin/refactor-main-cleanup-and-merge for review
git push origin refactor/main-cleanup-and-merge

# Step 9: After review, merge to main
git checkout main
git merge refactor/main-cleanup-and-merge
git push origin main
```

---

## README UPDATES NEEDED

After merge, update [README.md](README.md):

1. **Fix E2 canonical source** (already correct, but verify):
   - Point to `parrots/aa_fortu/multihead_analysis_graphs.py` line 1408
   - Note checkpoint set: step1, step1000, step5000, step7000, step10000, step100000, steplatest

2. **Add E2a (Appendix)**: Head Ablation Heatmaps
   - Script: `ablation_head_cycle_evolution.py`
   - Output: `head_ablation_heatmap_natural.png`, `head_ablation_heatmap_icl.png`

3. **Add E2b (Appendix)**: Cycle Prevalence Table
   - Script: `python scripts/aggregate_cycles.py && python scripts/write_table_from_aggregated.py`
   - Output: `cycle_statistics_by_model.tex`

4. **Add E6b (Appendix)**: Entropy Evolution (3-way categorization)
   - Script: `python entropy_3way_categorization.py`
   - Output: `entropy_categorical_evolution.png`

5. **Add Appendix Section**: Multi-Model Reproduction
   - OLMo: `experiments/olmo_attention_pipeline.py` + checkpoints guide
   - Apertus: `experiments/run_apertus_checkpoints.sh` + ablation setup
   - Llama: Per-model runners in `run_Llama_*.sh` (from chaotic merge)

6. **Add Appendix Section**: Top-p Mitigation via Head Policy
   - Script: `prepare_head_policy_library.py` → `plot_head_policy_matrix_results.py`
   - Output: `head_policy_mitigation_matrix.png`

---

## SUMMARY

| Category | Count | Status |
|---|---|---|
| Files to DELETE from main | ~130 | ❌ **New** — Exploratory scripts & docs |
| Files to MERGE from chaotic | ~75 | ✅ Identified — Appendix experiments |
| Files to SKIP from chaotic | ~150+ | ✅ Documented — Exploratory branches |
| Exploratory branches to archive | 2 | ⚠️ `cycle-attention-analysis/`, `untrained_repetition/` |
| README updates needed | 5+ sections | 📋 Pending |

**Result:** After deletions + merges, `main` will have:
- ✅ Complete paper reproducibility (all 7 main experiments)
- ✅ All appendix experiments (OLMo, Apertus, Llama, multi-model)
- ✅ Clean codebase free of exploratory dead-ends
- ✅ ~150 files removed, ~75 files added → net ~75 files added

