# OLMo Attention Analysis - Complete Guide

## Summary

This guide explains how to replicate the Pythia checkpoint-by-checkpoint attention analysis experiments with OLMo models.

### Original Pythia Experiments

**Purpose**: Track how attention patterns evolve across training checkpoints
**Data Generation**: `parrots/aa_fortu/ckpt_pipeline_main.py`
**Visualization**: `run_alluvial_dual.py` (creates alluvial plots)

**Pythia Checkpoints Analyzed**:
- step1 (very early training)
- step1000
- step5000  
- step10000
- step100000
- steplatest (final)

**Key Layer**: Layer 19/24 (79% depth)

### OLMo Adaptation

**Challenge**: OLMo models don't have publicly available intermediate checkpoints

**Solution**: Single-model analysis focusing on attention patterns without checkpoint evolution

## Scripts Created

### 1. Structure Inspection
**File**: `experiments/inspect_olmo_structure.py`
**Purpose**: Understand OLMo architecture and hook paths
**Key Finding**: OLMo uses `model.layers.{layer}` (not `gpt_neox.layers.{layer}`)

### 2. OLMo-Specific Analysis Pipeline  
**File**: `experiments/olmo_attention_pipeline.py`
**Purpose**: Adapted version of `ckpt_pipeline_main.py` for OLMo
**Key Changes**:
- Hook path: `model.layers.{layer}` instead of `gpt_neox.layers.{layer}`
- No revision parameter (no checkpoints)
- 16 layers instead of 24

### 3. SLURM Job Script
**File**: `experiments/slurm_olmo_attention.sh`
**Purpose**: Run attention analysis on SLURM cluster
**Configuration**:
- Partition: alien
- QoS: alien
- Memory: 64GB
- GPU: 1x A30
- Time: 4 hours

### 4. Master Launch Script
**File**: `experiments/run_olmo_attention_analysis.sh`
**Purpose**: Submit jobs for multiple layers
**Modes**:
- `single`: Run layer 12 only (default)
- `all`: Run all 16 layers
- `key`: Run layers 3, 7, 12, 15 (25%, 50%, 75%, 100% depth)

## Architecture Comparison

| Feature | Pythia-1.4b | OLMo-1B |
|---------|-------------|---------|
| Layers | 24 | 16 |
| Heads | 16 | 16 |
| Hidden Size | 2048 | 2048 |
| Vocab Size | 50304 | 50304 |
| Hook Pattern | `gpt_neox.layers.X` | `model.layers.X` |
| Checkpoints | 6 training steps | None available |
| Target Layer | 19 (79%) | 12 (75%) |

## Usage

### Quick Start (Single Layer)

Run layer 12 analysis (proportional to Pythia layer 19):

```bash
cd /home/mmahaut/projects/parrots
./experiments/run_olmo_attention_analysis.sh single
```

### Comprehensive Analysis (All Layers)

Run analysis on all 16 layers:

```bash
./experiments/run_olmo_attention_analysis.sh all
```

### Key Layers Only

Run analysis on representative layers:

```bash
./experiments/run_olmo_attention_analysis.sh key
```

### Manual Single Layer

```bash
sbatch experiments/slurm_olmo_attention.sh 12
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch output
tail -f logs/olmo_attention_*.out

# Check errors
tail -f logs/olmo_attention_*.err
```

## Expected Outputs

### Per Layer Output

Location: `outputs/olmo_attention/allenai/OLMo-1B-hf/layer_{L}/`

Files:
- `full_analysis_cyc4_ml32.out` - Analysis results
- `olmo_unexpected_heatmap_cyc4.png` - Visualization

### Console Output Format

```
layer 12 cycle count: 0.123
layer 12 natural heatmap: [0.05,0.12,0.08,...]  # 16 values (one per head)
layer 12 icl cycle count: 0.234
layer 12 icl heatmap: [0.03,0.09,0.11,...]
layer 12 no-cycle icl cycle count: 0.345
layer 12 no-cycle icl heatmap: [0.02,0.07,0.13,...]
layer 12 data index: [0,1,2,3,...]  # Datapoint indices
layer 12 repetition index: [5,12,23,...]  # Repeating datapoint indices
layer 12 no-cycle index: [8,15,29,...]  # No-cycle ICL indices
```

## Comparison with Pythia

### What Can Be Compared

✅ **Attention patterns at similar depth**
- Pythia layer 19/24 vs OLMo layer 12/16
- Both at ~75-79% model depth

✅ **Repetition behavior**
- Natural repetition rates
- ICL vs non-ICL patterns
- No-cycle ICL patterns

✅ **Head-wise analysis**
- Both models have 16 attention heads
- Can compare head-specific contributions

### What Cannot Be Compared Directly

❌ **Training evolution**
- Pythia has 6 checkpoints showing training progression
- OLMo only has final model

❌ **Checkpoint-to-checkpoint flows**
- Alluvial plots show how datapoints transition between checkpoints
- Not possible without intermediate OLMo checkpoints

## Alternative Approaches for Evolution Analysis

### Option 1: Model Family Comparison
Compare different OLMo sizes as "pseudo-checkpoints":
- OLMo-1B-hf (smaller)
- OLMo-7B-hf (larger)

Caveat: Different model sizes ≠ different training steps

### Option 2: Layer-by-Layer Comparison
Analyze how patterns change across OLMo layers:
- Early layers (0-5)
- Middle layers (6-10)
- Late layers (11-15)

Shows depth progression within model, not training progression

### Option 3: Contact Ai2 for Checkpoints
Request intermediate training checkpoints from Allen AI team

## Generating Alluvial-Style Visualizations

Even without checkpoints, we can create modified visualizations:

### Single-Model Alluvial
Show how datapoints flow between:
- Early layers → Middle layers → Late layers

### Cross-Model Comparison
If analyzing multiple OLMo variants:
- OLMo-1B → OLMo-7B (size progression)

## Next Steps

1. **Run initial analysis**:
   ```bash
   ./experiments/run_olmo_attention_analysis.sh single
   ```

2. **Validate output format matches Pythia**:
   - Check for natural heatmap values
   - Verify ICL and no-cycle ICL data
   - Confirm datapoint indices are captured

3. **Compare layer 12 results with Pythia layer 19**:
   - Load Pythia data from `outputs_multihead_full/EleutherAI/pythia-1.4b/steplatest/layer_19/`
   - Compare attention patterns
   - Analyze differences

4. **Scale to all layers** (if single layer succeeds):
   ```bash
   ./experiments/run_olmo_attention_analysis.sh all
   ```

5. **Create comparison visualizations**:
   - Side-by-side heatmaps
   - Layer-wise progression plots
   - OLMo vs Pythia comparisons

## Troubleshooting

### Job Fails Immediately
- Check logs: `cat logs/olmo_attention_*.err`
- Common issues:
  - Out of memory → increase `--mem` in SLURM script
  - Model download fails → check internet/HuggingFace access
  - Python env issues → verify conda environment

### No Output Generated
- Check if job completed: `squeue -u $USER`
- Verify output directory exists: `ls -la outputs/olmo_attention/`
- Check for errors in log files

### Results Look Wrong
- Compare with Pythia baseline
- Check if model loaded correctly (16 layers, 16 heads)
- Verify hook registration (should see "registered hooks" in output)

## Files Reference

```
experiments/
├── OLMO_ALLUVIAL_ADAPTATION.md          # Initial adaptation plan
├── OLMO_PYTHIA_COMPARISON.md            # Architecture comparison
├── inspect_olmo_structure.py            # Model structure inspector
├── olmo_attention_pipeline.py           # Main analysis script
├── slurm_olmo_attention.sh              # SLURM job script
├── run_olmo_attention_analysis.sh       # Master launcher
└── OLMO_ATTENTION_GUIDE.md             # This file

outputs/olmo_attention/
└── allenai/
    └── OLMo-1B-hf/
        ├── layer_0/
        ├── layer_1/
        ...
        └── layer_15/
```

## Questions Answered

1. ✅ **Can aa_fortu scripts be adapted for OLMo?**
   - Yes, with hook path changes

2. ✅ **How to find correct checkpoints?**
   - OLMo doesn't have intermediate checkpoints
   - Focus on single-model analysis instead

3. ✅ **What's the equivalent target layer?**
   - Layer 12/16 (75%) ≈ Pythia layer 19/24 (79%)

4. ✅ **Can we generate meaningful comparisons?**
   - Yes, comparing attention patterns at similar depths
   - No, not checkpoint evolution (requires intermediate models)
