# Intermediate Results, 2026-04-02

The main question here is not cycle detection by itself. It is how top-p sampling changes model behavior in the natural setting and in ICL.

## Current Run Status

- 32 top-level jobs completed
- 6 top-level jobs still running
- 15 top-level jobs were cancelled by an external Slurm action

The completed results are already enough for an interim read on the top-p question because all four model families have finished `p0.5`, `p0.9`, and `icl` runs.

## What The Data Says

Across all four models, the strongest separation is not between `p0.5` and `p0.9` on exact match. The clearer effect is on factuality and direct-follow behavior:

- `icl` has the highest factuality by a large margin in every model, with NLI factuality around 0.79 overall.
- `p0.5` keeps more factual output than `p0.9` overall, but still trails `icl` substantially.
- `p0.9` is the least factual setting overall and shows the biggest drop relative to `icl`.
- Exact match is consistently higher for both top-p settings than for `icl`, so raw match score alone would miss the main tradeoff.

## Summary Table

| Model | Condition | Exact match | NLI factual | Direct follow |
| --- | --- | ---: | ---: | ---: |
| Llama | p0.5 | 0.380 | 0.634 | 0.257 |
| Llama | p0.9 | 0.379 | 0.508 | 0.246 |
| Llama | icl | 0.287 | 0.788 | 0.000 |
| OLMo | p0.5 | 0.347 | 0.439 | 0.242 |
| OLMo | p0.9 | 0.335 | 0.089 | 0.217 |
| OLMo | icl | 0.290 | 0.791 | 0.000 |
| OPT | p0.5 | 0.328 | 0.410 | 0.247 |
| OPT | p0.9 | 0.330 | 0.102 | 0.220 |
| OPT | icl | 0.289 | 0.790 | 0.000 |
| Pythia-1.4B | p0.5 | 0.335 | 0.403 | 0.225 |
| Pythia-1.4B | p0.9 | 0.329 | 0.114 | 0.204 |
| Pythia-1.4B | icl | 0.290 | 0.790 | 0.000 |

Overall averages across the four models:

| Condition | Exact match | NLI factual | Direct follow |
| --- | ---: | ---: | ---: |
| p0.5 | 0.347 | 0.471 | 0.243 |
| p0.9 | 0.343 | 0.203 | 0.222 |
| icl | 0.289 | 0.790 | 0.000 |

## Interim Analysis

The practical story is:

1. `icl` is the most stable and factual regime.
2. `p0.5` is closer to `icl` than `p0.9` on factuality, but still clearly below it.
3. `p0.9` pushes the models furthest away from factual continuation, so if the goal is preserving natural behavior while sampling, it is the riskiest setting of the three.
4. The exact-match metric moves in the opposite direction from factuality, which means the top-p effect is not simply “more or less repetition”; it is a tradeoff between matching the target and remaining factually grounded.

## Interpretation For The Paper

The clean take-away is that top-p changes natural and ICL behavior differently:

- In the natural/top-p setting, the model can still match the target more often, especially at `p0.5`, but factual consistency drops as sampling becomes looser.
- In ICL, the model remains far more factual across the board, which suggests the in-context setup is acting as a stronger constraint than the sampling choice alone.
- The gap between `p0.5` and `p0.9` is modest for exact match, but large for factuality, so the paper should emphasize factual stability rather than raw match rate when discussing top-p.

## Relevant Paths

- [run_cycle_detection_for_new_models.py](run_cycle_detection_for_new_models.py)
- [outputs/mitigations/Llama_p0.5/slot_filling_results.csv](outputs/mitigations/Llama_p0.5/slot_filling_results.csv)
- [outputs/mitigations/Llama_p0.9/slot_filling_results.csv](outputs/mitigations/Llama_p0.9/slot_filling_results.csv)
- [outputs/mitigations/Llama_icl/slot_filling_results.csv](outputs/mitigations/Llama_icl/slot_filling_results.csv)
- [outputs/mitigations/OLMo_p0.5/slot_filling_results.csv](outputs/mitigations/OLMo_p0.5/slot_filling_results.csv)
- [outputs/mitigations/OLMo_p0.9/slot_filling_results.csv](outputs/mitigations/OLMo_p0.9/slot_filling_results.csv)
- [outputs/mitigations/OLMo_icl/slot_filling_results.csv](outputs/mitigations/OLMo_icl/slot_filling_results.csv)
- [outputs/mitigations/OPT_p0.5/slot_filling_results.csv](outputs/mitigations/OPT_p0.5/slot_filling_results.csv)
- [outputs/mitigations/OPT_p0.9/slot_filling_results.csv](outputs/mitigations/OPT_p0.9/slot_filling_results.csv)
- [outputs/mitigations/OPT_icl/slot_filling_results.csv](outputs/mitigations/OPT_icl/slot_filling_results.csv)
- [outputs/mitigations/Pythia-1.4B_p0.5/slot_filling_results.csv](outputs/mitigations/Pythia-1.4B_p0.5/slot_filling_results.csv)
- [outputs/mitigations/Pythia-1.4B_p0.9/slot_filling_results.csv](outputs/mitigations/Pythia-1.4B_p0.9/slot_filling_results.csv)
- [outputs/mitigations/Pythia-1.4B_icl/slot_filling_results.csv](outputs/mitigations/Pythia-1.4B_icl/slot_filling_results.csv)