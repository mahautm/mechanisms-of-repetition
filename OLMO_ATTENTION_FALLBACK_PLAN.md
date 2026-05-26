# OLMo-1B Attention Fallback Analysis

## Objective
Reproduce the attention spread/newline analysis figures from Pythia-1.4b for OLMo-1B to verify that newline tokens are not causal for repetition mechanisms across different model architectures.

## Figures Being Reproduced

### 1. Natural vs No-Cycle Comparison
**Original:** `/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison/paper_figure_natural_vs_nocycle_comparison.png`

**Description:** Multi-panel comparison showing:
- Attention shifts when newlines removed (Natural vs No-Cycle-ICL)
- Semantic vs Structural token attention
- Heatmap of attention shifts
- Key token type correlation

**Key Finding (Pythia-1.4b):**
- Both repetitive (Natural) and non-repetitive (No-Cycle-ICL) sequences show similar attention redistribution
- When newlines removed, attention shifts to content words
- Newlines are NOT causal for repetition mechanisms

### 2. Natural vs ICL Clean Comparison
**Original:** `/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison/paper_figure_natural_vs_icl_clean.png`

**Description:** Publication-ready side-by-side bar chart showing:
- Attention shifts for key token types (Content Word, Function Word, Punctuation, etc.)
- Direct comparison between Natural (repetitive) and ICL (in-context learning) sequences
- Statistical significance markers for major differences

## Modified Scripts

### 1. `compare_attention_fallback_natural_vs_nocycle.py`
**Changes Made:**
- Added `--model_name` parameter (default: "EleutherAI/pythia-1.4b")
- Added automatic layer selection at 75% depth if `--target_layer` not specified
- Updated output directory naming to include model name
- Dynamic model loading based on parameter

**Usage:**
```bash
python compare_attention_fallback_natural_vs_nocycle.py \
    --model_name "allenai/OLMo-1B-hf" \
    --n_samples 100 \
    --output_dir "./plots/attention_fallback_comparison_allenai_OLMo-1B-hf"
```

### 2. `create_paper_comparison_figure.py`
**Changes Made:**
- Added `--model_name` parameter for figure labeling
- Added `--results_dir` parameter to specify input directory
- Updated output filenames to include model identifier

**Usage:**
```bash
python create_paper_comparison_figure.py \
    --model_name "allenai/OLMo-1B-hf" \
    --results_dir "./plots/attention_fallback_comparison_allenai_OLMo-1B-hf"
```

## Analysis Pipeline

### Step 1: Generate Attention Fallback Data
The `compare_attention_fallback_natural_vs_nocycle.py` script:

1. **Loads OLMo-1B model** with automatic layer selection (layer 12 = 75% depth)
2. **Generates sequences:**
   - Natural (repetitive): Model-generated text with natural repetition patterns
   - No-Cycle-ICL: Non-repetitive sequences from in-context learning setup
3. **For each sequence:**
   - Computes baseline attention distribution (with newlines)
   - Removes newline tokens
   - Computes attention distribution without newlines
   - Classifies tokens into categories: CONTENT_WORD, FUNCTION_WORD, NEWLINE, PUNCTUATION, etc.
4. **Analyzes attention shifts:**
   - Calculates change in attention to each token type
   - Aggregates across all heads at target layer
   - Compares Natural vs No-Cycle-ICL patterns

**Output Files:**
- `attention_fallback_comparison_results.json`: Raw data with per-sequence results
- `paper_figure_natural_vs_nocycle_comparison.png`: Multi-panel visualization
- `attention_fallback_natural_vs_nocycle_report.md`: Detailed statistical report

### Step 2: Create Publication Figure
The `create_paper_comparison_figure.py` script:

1. **Loads results** from Step 1
2. **Aggregates statistics** across all sequences
3. **Creates clean comparison** focusing on key token types
4. **Generates publication-ready figure** with proper styling

**Output File:**
- `paper_figure_natural_vs_icl_clean_allenai_OLMo-1B-hf.png`

## Token Classification System

Uses NLTK for linguistic analysis:

### Semantic Tokens (Meaning-bearing)
- **CONTENT_WORD**: Nouns, verbs, adjectives, adverbs (POS-tagged)
- **FUNCTION_WORD**: Determiners, prepositions, conjunctions (POS-tagged + stopwords)
- **PROGRAMMING**: Code keywords (`def`, `class`, `import`, `=`)

### Structural Tokens (Non-semantic)
- **NEWLINE**: Line breaks (`Ċ`, `\n`)
- **SENTENCE_END**: Sentence terminators (`.`, `!`, `?`)
- **PUNCTUATION**: Internal punctuation (`,`, `;`, `:`)
- **BRACKET**: Grouping symbols (`(`, `)`, `[`, `]`)
- **NUMBER**: Numeric tokens

### Other
- **OTHER**: Special tokens, BOS, unknown

## Current Job Status

**Job ID:** 1841491  
**Status:** Running on node039  
**Started:** 2025-11-20 18:30:48 CET  

**Expected Runtime:** 2-4 hours for 100 samples per sequence type

**Output Location:**
```
/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/plots/attention_fallback_comparison_allenai_OLMo-1B-hf/
```

**Log Files:**
- Output: `logs/olmo_attention_fallback_1841491.out`
- Error: `logs/olmo_attention_fallback_1841491.err`

## Expected Results

Based on Pythia-1.4b findings, we expect:

1. **Similar Attention Redistribution:**
   - Both Natural and No-Cycle-ICL sequences should show comparable attention shifts
   - Primary fallback target: CONTENT_WORD tokens

2. **Newline Token Evidence:**
   - Small negative shift for NEWLINE category (attention loss when removed)
   - Attention redistributes mainly to semantic content
   - No major difference between repetitive vs non-repetitive sequences

3. **Structural Pattern:**
   - Minimal attention to PUNCTUATION, BRACKET tokens
   - Slight increase to FUNCTION_WORD tokens
   - Strong increase to CONTENT_WORD tokens

4. **Key Conclusion:**
   - Newlines are NOT causal for repetition mechanisms
   - Attention patterns remain consistent across OLMo-1B architecture
   - Validates findings from Pythia-1.4b across different model families

## Comparison with Original Pythia-1.4b Results

### Pythia-1.4b (Layer 19, 75% depth)
- Natural sequences: CONTENT_WORD shift ≈ +62pp
- No-Cycle-ICL: CONTENT_WORD shift ≈ +60pp  
- Difference: ~2pp (minimal)
- **Conclusion:** Newlines not causal

### OLMo-1B (Layer 12, 75% depth)
- Results pending...
- Expected: Similar pattern to Pythia
- Will confirm robustness across architectures

## Files Modified

1. `/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/compare_attention_fallback_natural_vs_nocycle.py`
   - Added model parameter support
   - Dynamic layer selection
   
2. `/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/create_paper_comparison_figure.py`
   - Added model parameter support
   - Flexible input/output paths

3. `/home/mmahaut/projects/parrots/cycle-attention-analysis/experiments/phase3_newline_causality/run_olmo_attention_fallback.sh` (NEW)
   - SLURM submission script
   - Automated two-step pipeline

## Next Steps

1. **Wait for job completion** (~2-4 hours)
2. **Review generated figures** in output directory
3. **Compare with Pythia-1.4b results** to verify consistency
4. **Add to rebuttal document** if findings confirm hypothesis
5. **Optional:** Repeat for instruction-tuned OLMo-2-0425-1B-Instruct

## Monitor Progress

```bash
# Check job status
squeue -j 1841491

# View output log
tail -f logs/olmo_attention_fallback_1841491.out

# Check for errors
tail -f logs/olmo_attention_fallback_1841491.err

# Check job accounting after completion
sacct -j 1841491 --format=JobID,State,Elapsed,MaxRSS
```
