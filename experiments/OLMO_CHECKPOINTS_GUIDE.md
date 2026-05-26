# OLMo Checkpoint-by-Checkpoint Analysis - UPDATED

## 🎉 Great News: OLMo HAS Intermediate Checkpoints!

OLMo-1B has **335 intermediate checkpoints** available on HuggingFace, released every 1000 training steps!

## Available Checkpoints

### Format
Checkpoints use the naming convention: `step{N}-tokens{T}B`
- Example: `step20000-tokens84B` means 20,000 training steps, 84 billion tokens seen

### Recommended Checkpoint Selection (6 checkpoints)

To match Pythia's 6-checkpoint pattern, we select checkpoints spread across training:

| # | Checkpoint | Training Steps | Tokens Seen | % Training |
|---|------------|----------------|-------------|------------|
| 1 | `step20000-tokens84B` | 20,000 | 84B | ~3% (early) |
| 2 | `step100000-tokens419B` | 100,000 | 419B | ~14% |
| 3 | `step200000-tokens839B` | 200,000 | 839B | ~27% |
| 4 | `step350000-tokens1468B` | 350,000 | 1.47T | ~47% |
| 5 | `step500000-tokens2097B` | 500,000 | 2.10T | ~68% |
| 6 | `step738020-tokens3095B` | 738,020 | 3.10T | 100% (final) |

### Comparison with Pythia Checkpoints

| Pythia (1.4b) | OLMo (1B) Equivalent |
|---------------|----------------------|
| step1 | step20000-tokens84B (earliest available) |
| step1000 | step100000-tokens419B |
| step5000 | step200000-tokens839B |
| step10000 | step350000-tokens1468B |
| step100000 | step500000-tokens2097B |
| steplatest | step738020-tokens3095B |

## Updated Scripts

### 1. Modified SLURM Script with Checkpoint Support

The `slurm_olmo_attention.sh` script now accepts both layer and checkpoint parameters:

```bash
sbatch experiments/slurm_olmo_attention.sh LAYER CHECKPOINT
```

Example:
```bash
# Run layer 12 at step 100k
sbatch experiments/slurm_olmo_attention.sh 12 step100000-tokens419B

# Run layer 12 at final checkpoint
sbatch experiments/slurm_olmo_attention.sh 12 step738020-tokens3095B
```

### 2. Run All Checkpoints for a Layer

```bash
./experiments/run_olmo_checkpoints.sh 12
```

This will submit 6 jobs analyzing layer 12 across all recommended checkpoints.

### 3. Full Checkpoint × Layer Analysis

```bash
./experiments/run_olmo_full_checkpoint_analysis.sh
```

This runs the complete experiment: 6 checkpoints × target layer (layer 12) = 6 jobs.

## Output Structure

```
outputs/olmo_attention/
└── allenai/
    └── OLMo-1B/
        ├── step20000-tokens84B/
        │   └── layer_12/
        │       └── full_analysis_cyc4_ml32.out
        ├── step100000-tokens419B/
        │   └── layer_12/
        ├── step200000-tokens839B/
        │   └── layer_12/
        ├── step350000-tokens1468B/
        │   └── layer_12/
        ├── step500000-tokens2097B/
        │   └── layer_12/
        └── step738020-tokens3095B/
            └── layer_12/
```

## Generating Alluvial Plots

Once all checkpoints are analyzed, you can generate alluvial plots showing how attention patterns evolve:

```bash
python experiments/olmo_alluvial_plot.py \
    --model allenai/OLMo-1B \
    --layer 12 \
    --output-dir plots/olmo_alluvial/
```

This will create:
- Natural repetition alluvial plot
- ICL repetition alluvial plot  
- No-cycle ICL alluvial plot
- Dual comparison plots (like Pythia)

## Key Advantages

1. **True Checkpoint Evolution**: Can now track how OLMo's attention patterns evolve during training
2. **Direct Pythia Comparison**: 6 checkpoints → 6 checkpoints comparison
3. **Fine-Grained Analysis**: 335 total checkpoints available if you need more granularity

## Usage Examples

### Quick Test (Single Checkpoint + Layer)
```bash
sbatch experiments/slurm_olmo_attention.sh 12 step100000-tokens419B
```

### Full Alluvial Analysis (6 Checkpoints, Layer 12)
```bash
for checkpoint in step20000-tokens84B step100000-tokens419B step200000-tokens839B \
                  step350000-tokens1468B step500000-tokens2097B step738020-tokens3095B; do
    sbatch experiments/slurm_olmo_attention.sh 12 $checkpoint
done
```

### Monitor All Jobs
```bash
squeue -u $USER
tail -f logs/olmo_attention_*.out
```

## Next Steps

1. **Run checkpoint analysis** for layer 12 across all 6 checkpoints
2. **Verify output format** matches Pythia (data indices, repetition indices, heatmaps)
3. **Generate alluvial plots** using `run_alluvial_dual.py` adapted for OLMo
4. **Compare OLMo vs Pythia** checkpoint evolution patterns

## Notes

- OLMo checkpoints start at step 20,000 (not step 1 like Pythia)
- Final OLMo-1B checkpoint is at step 738,020 (3.1T tokens)
- Pythia steplatest is at ~143,000 steps (300B tokens)
- OLMo trained ~20x longer than Pythia!
