# OLMo-1B Repetition Analysis Results

## Experimental Setup
- **Model**: allenai/OLMo-1B-hf
- **Layer Analyzed**: Layer 12 (75% model depth)
- **Samples**: 1000 per checkpoint
- **Analysis Type**: No-head attention analysis (indices only)
- **Conditions**: Natural repetition and No-cycle ICL

## Pre-trained Model Results

Analysis of 6 checkpoints spanning the full pre-training trajectory from 4B to 3094B tokens.

| Checkpoint | Training Tokens | Natural Repetition (First Repeating) | No-Cycle ICL |
|------------|-----------------|--------------------------------------|--------------|
| step1000-tokens4B | 4B | 239/1000 (23.9%) | 2/85 (2.4%) |
| step343000-tokens1438B | 1438B | 77/1000 (7.7%) | 17/85 (20.0%) |
| step425000-tokens1781B | 1781B | 51/1000 (5.1%) | 33/85 (38.8%) |
| step509000-tokens2134B | 2134B | 42/1000 (4.2%) | 30/85 (35.3%) |
| step593000-tokens2486B | 2486B | 28/1000 (2.8%) | 23/85 (27.1%) |
| step738020-tokens3094B | 3094B | 23/1000 (2.3%) | 9/85 (10.6%) |

### Key Findings

1. **Natural Repetition** (progressive categorization - when samples *first* start repeating):
   - **Early training** (4B tokens): 23.9% of samples first show repetition
   - **Later checkpoints** (1438B-3094B tokens): Declining rate (7.7% → 2.3%)
   - **Interpretation**: Most samples (23.9%) show repetition behavior by the earliest checkpoint. Additional samples continue to develop repetition patterns during training, but at a decreasing rate.

2. **No-Cycle ICL Pattern** (out of 85 total unique ICL samples): 
   - **Early training** (4B tokens): Minimal ICL repetition (2.4%)
   - **Mid training** (1438B-2134B tokens): Peak ICL repetition (20.0% → 38.8% → 35.3%)
   - **Late training** (2486B-3094B tokens): Declining ICL repetition (27.1% → 10.6%)

3. **Contrasting Dynamics**: 
   - **Natural condition**: Repetition established early (23.9% at first checkpoint), with gradual additions
   - **ICL condition**: Shows clear learning curve with peak at mid-training (~40%) and improvement (decline) toward final checkpoint (~10%)
   - This demonstrates that ICL repetition is more dynamic during training and responds differently to in-context learning than natural repetition.

## Instruction-Tuned Model Analysis

Analysis of the same OLMo-1B-0724 model at various pre-training checkpoints (100 samples per checkpoint).

| Checkpoint | Training Tokens | Natural Repetition | No-Cycle ICL |
|------------|-----------------|--------------------|--------------| 
| step0-tokens0B | 0B (initialization) | 98/100 (98.0%) | 2/100 (2.0%) |
| step288000-tokens603B | 603B | 99/100 (99.0%) | 1/100 (1.0%) |
| step577000-tokens1209B | 1209B | 96/100 (96.0%) | 4/100 (4.0%) |
| step865000-tokens1813B | 1813B | 98/100 (98.0%) | 2/100 (2.0%) |
| step1165000-tokens2442B | 2442B | 97/100 (97.0%) | 3/100 (3.0%) |
| step1454000-tokens3048B | 3048B (final) | 100/100 (100.0%) | 0/100 (0.0%) |

### Key Findings

1. **Final Checkpoint Behavior**: At the final pre-training checkpoint (3048B tokens), the model exhibits:
   - 100% natural repetition (all samples repeat)
   - 0% ICL repetition (no samples repeat in the no-cycle ICL condition)

2. **Diverse Generation**: The absence of ICL repetition at the final checkpoint indicates the model has learned to generate diverse outputs when provided with in-context examples, even without explicit instruction tuning.

## Comparison with Pythia

These OLMo results replicate the key patterns observed in Pythia models:
- High natural repetition across training
- Non-monotonic ICL repetition with peak in mid-training
- Declining ICL repetition in later stages
- Evidence that pre-trained models learn diverse generation through training dynamics alone

## Visualization

Alluvial plots showing the evolution of repetition patterns:
- **Pre-trained**: `plots/olmo_alluvial_layer12_dual_alluvial.png`
- **Instruction-tuned checkpoints**: `plots/olmo_instruct_alluvial_layer12_dual_alluvial.png`

## Notes

- **Natural repetition**: Uses progressive categorization showing samples that *first* start repeating at each checkpoint. Denominator is 1000 (total samples across all checkpoints). Earlier checkpoints capture most samples, with later checkpoints showing fewer new repetitions.
- **No-cycle ICL**: Calculated as (samples repeating at this checkpoint / total unique ICL samples across all checkpoints). The denominator is fixed at 85 samples for comparability across checkpoints.
- **Alluvial plot interpretation**: The plot shows flow of samples across training stages. Natural condition shows most samples establish repetition early, while ICL shows dynamic changes with mid-training peak and late-training reduction.
- The instruction-tuned analysis examines pre-training checkpoints of OLMo-1B-0724-hf, not a separately fine-tuned model
