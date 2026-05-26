# Head Intervention Experiment: Selective Control of Natural Repetition Without Harming ICL

## Experiment Philosophy

The reviewer-facing causal question is:

**Can we reduce a specific behavior (natural repetition) without harming ICL behavior?**

This experiment uses attention-head interventions learned from ablation results to test selective control, not just overall mitigation.

Top-p is a strong baseline that may already satisfy this criterion. The head-intervention objective is to provide:
- causal localization (which heads drive selective effects), and
- a potentially more targeted control policy than global sampling changes.

---

## Primary Claim (Reviewer-Ready)

We will evaluate selective control as a constrained causal objective:

- **Efficacy target (natural):** reduce natural cyclical rate by at least $\delta_{nat}$
- **Safety target (ICL):** keep ICL change within a non-inferiority margin $\epsilon_{icl}$

Concrete default thresholds:
- $\delta_{nat} = 20\%$ relative reduction vs baseline
- $\epsilon_{icl} = 2\%$ absolute change in ICL cyclical rate (or an equivalent ICL quality metric margin)

Any intervention that satisfies both is considered successful.

---

## Research Questions

1. **Can we selectively control natural repetition while preserving ICL?**
   - Which interventions satisfy both efficacy and safety constraints?
   - Are there Pareto-optimal policies that beat or match top-p?

2. **Which heads enable selectivity?**
   - Which heads strongly affect natural repetition but minimally affect ICL?
   - Are selective heads stable across checkpoints/models?

3. **How do selective interventions alter cycle dynamics?**
   - Do they delay onset, reduce cycle tightness, or shorten cycle persistence in natural?
   - Do they preserve ICL structure compared to top-p?

---

## Experimental Design

### Phase 1: Head Selection From Existing Ablations
**Objective**: Convert ablation tables into selective candidate head sets.

**Method**:
- Use condition-aware ablation deltas:
  - $\Delta rep_{natural,h}$
  - $\Delta rep_{icl,h}$
- Build selectivity scores:
  - Benefit: $B_h = \Delta rep_{natural,h}$
  - Collateral: $C_h = |\Delta rep_{icl,h}|$
  - Selectivity: $S_h = B_h - \lambda C_h$
- Rank heads by $S_h$ and keep sparse sets (top 1-4 heads).
- Evaluate separately for:
  - natural suppressor heads (amplify), and
  - natural promoter heads (suppress/ablate).

**Output**:
- Candidate intervention sets ranked by selectivity score
- Pareto frontier seed configurations

### Phase 2: Intervention Policies
**Objective**: Define selective policies and compare against top-p.

**Intervention Strategies**:

#### Strategy A: Selective Head Amplification
- Amplify natural suppressor heads: `head_output *= α`
  - $\alpha \in \{1.1, 1.25, 1.5, 2.0\}$
  - Hypothesis: natural repetition decreases while ICL remains stable

#### Strategy B: Selective Head Suppression
- Zero natural promoter heads: `head_output *= 0`
  - Hypothesis: natural repetition decreases with bounded ICL collateral

#### Strategy C: Hybrid Top-p + Head Policy
- Combine low-intensity top-p (for example $p=0.9$) with sparse head policy
- Hypothesis: better natural reduction at equal or lower ICL harm than either method alone

### Phase 3: Evaluation Protocol (Constrained Causal Test)
**Objective**: Test selective control and non-inferiority.

**Contexts to Compare**:

| Arm | Setup | Purpose |
|---------|-------|-------------------|
| **Baseline** | Greedy, no intervention | Reference |
| **Top-p** | Top-p only (0.5 and 0.9) | Operational baseline |
| **Head policy** | Selected sparse head intervention | Causal-selective policy |
| **Hybrid** | Top-p + selected head intervention | Combined control |

**Metrics to Extract**:

Primary:
- `delta_nat_cyclical_rate_vs_baseline`
- `delta_icl_cyclical_rate_vs_baseline`

Secondary:
- `delta_nat_mean_cycle_length_vs_baseline`
- `delta_icl_mean_cycle_length_vs_baseline`
- `cycle_onset_position`, `cycle_tightness`, `cycle_delay`, `cycle_consistency`
- throughput and efficiency (reduction per second)

Decision rule:
- Accept configuration if:
  - natural reduction meets $\delta_{nat}$, and
  - ICL change is within $\epsilon_{icl}$.

### Phase 4: Pareto and Non-Inferiority Analysis

**Outputs**:
1. **Pareto frontier**
   - axes: natural reduction vs ICL harm
   - points: top-p, head-only, hybrid configurations

2. **Non-inferiority table**
   - for each configuration: pass/fail against $(\delta_{nat}, \epsilon_{icl})$

3. **Selective head card**
   - per-head $B_h$, $C_h$, $S_h$, and stability across checkpoints

4. **Qualitative audit set**
   - side-by-side baseline vs top-p vs head-policy generations for representative prompts

---

## Implementation Plan

### Step 1: Prepare Data
- Use existing Pile minipile 512-sample corpus (seed=42)
- Pre-compute natural generation outputs with cycle labels
- Stratify into: non-cyclic (cycle_length=0), cyclic (cycle_length>0)

### Step 2: Head Probe (Optional if ablation data exists)
- Reuse existing condition-aware ablation outputs
- Compute $B_h$, $C_h$, and $S_h$
- Build top-k sparse candidate sets with greedy forward selection (k <= 4)

### Step 3: Build Intervention Runner
- Create `run_head_intervention_forced_cycles.py`
- Implement `create_head_amplification_hook()` (multiply head outputs)
- Implement `create_head_suppression_hook()` (zero head outputs)
- Generate in four arms: baseline, top-p, head-only, hybrid
- Track per-configuration outcome on both natural and ICL splits

### Step 4: Extract Metrics
- Enhance `detect_cycle()` to return onset_position, tightness, delay
- Build per-sample metric table for natural and ICL
- Aggregate by arm and by configuration
- Compute pass/fail against $(\delta_{nat}, \epsilon_{icl})$

### Step 5: Visualize & Interpret
- Plot Pareto frontier (natural gain vs ICL harm)
- Create non-inferiority summary table
- Compare against top-p as baseline method

---

## Expected Outcomes

### Success Case
- At least one head-policy (or hybrid) configuration passes non-inferiority constraints.
- Head-policy lands on Pareto frontier and is competitive with top-p.

### Partial Success
- Top-p remains best operationally, but head-policy identifies causal loci with interpretable selectivity.

### Failure Case
- Any natural gain requires unacceptable ICL degradation.
- Conclusion: selective control may require broader mechanisms than sparse head interventions.

---

## Key Insights We're After

1. **Selectivity**: Can we reduce natural repetition while preserving ICL?
2. **Causal localization**: Which heads provide this selectivity?
3. **Comparative value**: Does head-policy match or improve on top-p tradeoffs?
4. **Robustness**: Are selective heads stable across checkpoints/models?
5. **Mechanistic interpretation**: Are selective effects explained by onset/tightness shifts?

---

## Deliverables

| Deliverable | Format | Purpose |
|--|--|--|
| **Selective Head Ranking** | CSV | Per-head benefit/collateral/selectivity scores |
| **Intervention Runner** | Python script | `run_head_intervention_forced_cycles.py` |
| **Metric Dataset** | CSV | Per-sample metrics across baseline/top-p/head/hybrid arms |
| **Pareto + Non-Inferiority Figure** | PDF/PNG | Natural reduction vs ICL harm comparison |
| **Reviewer-Facing Report** | Markdown | Constrained causal claim with CIs and pass/fail table |

---

## Computational Requirements

- **GPU Memory**: ~4-8GB (similar to mitigation experiments, may need batches of 2-4)
- **Estimated Runtime**: 2-3 hours (512 samples × 5 contexts × 4 interventions = 10240 generations)
- **Models**: facebook/opt-1.3b, EleutherAI/pythia-1.4b, allenai/OLMo-1B-hf, meta-llama/Llama-3.2-1B
- **Cluster**: Use `srun` with `--partition=alien --qos=alien` per workspace guidelines

---

## Next Steps

1. ✅ Reframe objective as constrained selective control
2. ⏳ Add ablation-to-policy scorer ($B_h, C_h, S_h$)
3. ⏳ Run baseline/top-p/head/hybrid on same prompts
4. ⏳ Build Pareto + non-inferiority tables
5. ⏳ Write reviewer-facing conclusion
