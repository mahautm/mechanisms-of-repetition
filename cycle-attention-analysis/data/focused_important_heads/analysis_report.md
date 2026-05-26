# Important Attention Heads Analysis Report
## Summary
- Analysis based on contrast values at cycle size 3
- Template size: 32
- Top 10 heads identified for focused analysis
- Sequence types considered: natural, no-cycle ICL

## Top Important Heads
| Rank | Layer | Head | Contrast Value | Sequence Type | Raw Contrast |
|------|-------|------|----------------|---------------|--------------|
|  1 | 23 | 10 | 1.20e-05 | no_cycle_icl | -1.20e-05 |
|  2 | 17 | 10 | 5.02e-06 | no_cycle_icl | -5.02e-06 |
|  3 |  7 |  0 | 4.83e-06 | no_cycle_icl | +4.83e-06 |
|  4 | 15 | 14 | 4.76e-06 | no_cycle_icl | +4.76e-06 |
|  5 |  9 | 13 | 4.63e-06 | no_cycle_icl | +4.63e-06 |
|  6 | 11 |  1 | 4.58e-06 | no_cycle_icl | +4.58e-06 |
|  7 | 10 |  1 | 4.58e-06 | no_cycle_icl | +4.58e-06 |
|  8 |  8 | 14 | 4.23e-06 | no_cycle_icl | -4.23e-06 |
|  9 | 10 | 14 | 4.21e-06 | no_cycle_icl | -4.21e-06 |
| 10 | 19 | 10 | 4.19e-06 | no_cycle_icl | +4.19e-06 |

## Analysis Details
- **Contrast Path**: /home/mmahaut/projects/parrots/outputs_multihead_full/EleutherAI/pythia-1.4b/steplatest
- **Cycle Data Path**: /home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/cycle_evolution_parametric/cycles_4/steplatest
- **Total Layers Processed**: 24
- **Heads per Layer**: 16

## Layer Distribution
- **Layer 7**: 1 head
- **Layer 8**: 1 head
- **Layer 9**: 1 head
- **Layer 10**: 2 heads
- **Layer 11**: 1 head
- **Layer 15**: 1 head
- **Layer 17**: 1 head
- **Layer 19**: 1 head
- **Layer 23**: 1 head

## Generated Files
- `natural_focus_evolution_important_heads.png`: Focus token evolution for natural sequences
- `natural_attention_evolution_important_heads.png`: Attention distribution evolution for natural sequences
- `no_cycle_icl_focus_evolution_important_heads.png`: Focus token evolution for no-cycle ICL sequences
- `no_cycle_icl_attention_evolution_important_heads.png`: Attention distribution evolution for no-cycle ICL sequences