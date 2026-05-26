# ICL vs. Natural Condition Ablation: Implementation Plan

## Problem We're Solving

The original ablation pipeline (currently running on an extensive matrix) **only tests on generic text (minipile)**. 
This misses a critical distinction from earlier analyses:

- **Natural Repetition**: Model generates cycles on plain text → inherent architectural tendencies
- **ICL Repetition**: Model generates cycles when given pattern examples → learned in-context adaptation

**Why it matters**: Heads may have *different* causal roles in each condition, revealing mechanistic specialization.

## Three Implementation Tiers

### ✅ ALREADY DONE (Current Extensive Matrix)
```
- Natural condition only
- 46 layer-runs across Pythia-70m (6L), Pythia-1.4b (24L), OLMo-1B (16L)
- 6 checkpoints per model × 8-32 heads per layer
- Currently: Pythia-70m complete, Pythia-1.4b in progress
```

### 🔧 TIER 1: Quick Validation (Test Pipeline)
```
New script: ablation_head_cycle_icl_natural.py
- Test: Pythia-70m L0 only
- Conditions: Natural + ICL  
- Samples: 30 per condition (halves compute vs. current)
- Checkpoints: steplatest only (cuts 6x overhead)
- Heads: 2-3 sample heads per layer
- Runtime: ~30 min on GPU
- Purpose: Validate pipeline, verify effect sizes make sense
```

**Run now while extensive matrix continues:**
```bash
srun --partition=alien --qos=alien --exclude=node044 --mem=24G bash scripts/test_ablation_icl_natural.sh
```

### 🚀 TIER 2: Full Pythia-70m (Natural + ICL)
```
If TIER 1 validation passes:
- Model: Pythia-70m (6 layers × 8 heads)
- Conditions: Natural + ICL (2× overhead)
- Samples: 60 per condition (matches current ablation)
- Checkpoints: All 6 (step1, step1000, ..., steplatest)
- Total runs: 6 layers × 8 heads × 6 checkpoints × 2 conditions = 576 runs
- Estimated time: ~2-3 days
- Post-analysis: condition_analysis.csv + visualizations + insights
```

### 🔥 TIER 3: Full Extended Matrix (Natural + ICL)
```
If results from TIER 2 show significant condition-specific effects:
- Models: Pythia-70m (6L) + Pythia-1.4b (24L) + OLMo-1B (16L) = 46 layers
- Conditions: Natural + ICL (2× overhead)
- All checkpoints: 6 per model
- Total runs: ~46 × 8-32 heads × 6 checkpoints × 2 conditions = 10,000+ runs
- Estimated time: 1-2 weeks
- Output: Condition-specific effect matrices across all models
```

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Scripts created | ✅ | `ablation_head_cycle_icl_natural.py` + analysis + validation |
| Extensive natural-only matrix | 🔄 Running | Pythia-70m (6/6 done), Pythia-1.4b (~2/24 running) |
| Tier 1 validation ready | ⏳ Pending | Need your go-ahead to queue + test |
| Tier 2 full Pythia-70m | ⏳ Blocked | Waiting for Tier 1 validation results |

## Recommendation

**Start with Tier 1 immediately** (~30 min validation):
- Confirms ICL prompts work (detect cycles differently from natural?)
- Validates that condition-specific effects are significant
- If results look good → queue Tier 2 (Pythia-70m full) overnight
- If condition effects are weak → stick with natural-only, save 2 weeks

**Result of Tier 1 will guide Tier 2/3 scope.**

---

## Key Questions to Answer After Tier 1

1. **Do ICL and natural conditions show different effect profiles?**
   - If yes: condition-specific heads exist → Tier 2 justified
   - If no: safe to focus on natural condition (current path)

2. **Are ICL effects larger or smaller than natural?**
   - Larger: ICL conditions enable stronger head control (interesting!)
   - Smaller: heads more robust to task specifics (also interesting)

3. **What's the overlap between top natural and top ICL heads?**
   - High overlap: general-purpose heads
   - Low overlap: specialized heads per condition

---

## Files Generated

- `ablation_head_cycle_icl_natural.py` — Main pipeline (ICL + natural conditions)
- `analyze_ablation_icl_natural.py` — Comparison analysis + visualization
- `scripts/test_ablation_icl_natural.sh` — Validation test
- `plots/ablation_icl_natural_analysis/` — Output directory (created on first run)

## Next Steps

1. **Your decision**: Tier 1 validation with current cluster capacity? (Y/N)
2. **If yes**: I'll queue the validation test and monitor for ~30 min
3. **Based on results**: Decide if Tier 2 (full Pythia-70m) is worth 2-3 days
4. **Parallel**: Continue monitoring extensive natural-only matrix for completion

---

**TL;DR**: We've been missing the ICL vs. Natural distinction. Built full pipeline to test it.
Quick 30-min validation will tell us if condition-specific effects are worth weeks of compute.
What should we do?
