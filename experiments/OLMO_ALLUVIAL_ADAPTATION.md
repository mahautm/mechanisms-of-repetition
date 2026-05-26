# OLMo Alluvial Plot Adaptation Plan

## Summary of Original Experiments

### 1. Data Generation Scripts
- **Main script**: `parrots/aa_fortu/ckpt_pipeline_main.py`
  - Loads model with transformer-lens HookedTransformer
  - Runs attention contrast analysis across layers
  - Extracts:
    - Natural repetition patterns (cycle 0)
    - ICL repetition patterns
    - No-cycle ICL patterns
    - Datapoint indices and repetition indices per checkpoint

- **Output format**: Per checkpoint per layer
  - Location: `outputs_multihead_full/EleutherAI/pythia-1.4b/{checkpoint}/layer_{layer}/`
  - Files: `full_analysis_cyc{N}_ml{MAX_LENGTH}.out`
  - Contains:
    - `layer {L} data index: [...]` - all datapoints
    - `layer {L} repetition index: [...]` - repeating datapoints
    - `layer {L} no-cycle index: [...]` - no-cycle ICL datapoints

### 2. Plot Generation Script
- **Script**: `run_alluvial_dual.py`
  - Creates beautiful dual alluvial plots
  - Left subplot: ICL data
  - Right subplot: Natural data
  - Shows progressive categorization across checkpoints

### 3. Pythia Checkpoints Used
```python
checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest']
```

### 4. Key Layer Analyzed
- **Layer 19** (out of 24 total layers for pythia-1.4b)

## OLMo Adaptation Assessment

### Easy Adaptations ✅

1. **Model Loading**
   - Current code uses `AutoModelForCausalLM.from_pretrained(model_name, revision=revision)`
   - OLMo models work with same API
   - Just need to change model_name to `allenai/OLMo-1B-hf` or `allenai/OLMo-7B-hf`

2. **HookedTransformer Integration**
   - transformer-lens supports OLMo architecture
   - Same hook patterns work (`gpt_neox.layers.{layer}` → OLMo equivalent)

3. **Data Processing**
   - Uses generic text dataset
   - No model-specific data format
   - Tokenizer handled via AutoTokenizer

### Challenging Adaptations ⚠️

1. **No Training Checkpoints**
   - Pythia has checkpoints: step1, step1000, step5000, step10000, step100000, steplatest
   - OLMo models are typically released as:
     - Final trained model only
     - No intermediate checkpoints publicly available
   - **Solution**: Use OLMo family members as "pseudo-checkpoints":
     - OLMo-1B-hf (smaller, earlier-stopped equivalent)
     - OLMo-7B-hf (larger, more trained equivalent)
     - Or: Focus on single-model analysis without checkpoint evolution

2. **Layer Architecture Differences**
   - Pythia-1.4b: 24 layers
   - OLMo-1B: 16 layers
   - OLMo-7B: 32 layers
   - **Solution**: Analyze proportional layer (e.g., layer 12/16 for OLMo-1B ≈ layer 19/24 for Pythia)

3. **Hook Path Names**
   - Pythia uses: `gpt_neox.layers.{layer}`
   - OLMo might use different naming
   - **Solution**: Inspect OLMo model structure first

## Recommended Approach

### Option 1: Single OLMo Model Analysis (QUICK)
**What**: Analyze OLMo-1B without checkpoint evolution
**Time**: 1-2 days
**Steps**:
1. Run `ckpt_pipeline_main.py` on OLMo-1B for all layers
2. Compare attention patterns to Pythia-1.4b at similar proportional layers
3. Generate single-checkpoint visualizations

### Option 2: OLMo Family Comparison (MODERATE)
**What**: Compare OLMo-1B vs OLMo-7B as "pseudo-checkpoints"
**Time**: 3-5 days
**Steps**:
1. Run analysis on both OLMo-1B and OLMo-7B
2. Create modified alluvial plot with 2 "checkpoints"
3. Focus on whether patterns emerge similarly

### Option 3: Full Checkpoint Recreation (DIFFICULT)
**What**: Try to find/access OLMo intermediate checkpoints
**Time**: Unknown (may not be possible)
**Steps**:
1. Contact Ai2 for intermediate checkpoints
2. If available, run full pipeline like Pythia
3. Generate full alluvial plots

## Adaptation Steps (Option 1 - Recommended)

### Step 1: Inspect OLMo Architecture
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
print(model)
# Find correct layer path for hooks
```

### Step 2: Modify Hook Paths
Update `ckpt_pipeline_main.py` or create `ckpt_pipeline_olmo.py`:
```python
# Instead of: f"gpt_neox.layers.{str(single_lens)}"
# Use OLMo equivalent (TBD after inspection)
hooked_model = HookedModel(model, layer=f"model.layers.{str(single_lens)}")
```

### Step 3: Create OLMo SLURM Script
Based on `scripts/run_full_multihead_analysis.sh`:
```bash
MODEL_NAME="allenai/OLMo-1B-hf"
MAX_LAYERS=16  # OLMo-1B has 16 layers
TARGET_LAYER=12  # Proportional to pythia's layer 19
# No revision/checkpoint needed
```

### Step 4: Run Analysis
```bash
sbatch scripts/run_olmo_attention_analysis.sh
```

### Step 5: Generate Comparison Plot
Modify `run_alluvial_dual.py` or create new comparison script

## File Changes Needed

1. **New Script**: `experiments/olmo_attention_analysis.py`
   - Adapted version of `ckpt_pipeline_main.py`
   - OLMo-specific hook paths
   - Single model (no checkpoints)

2. **New SLURM Script**: `experiments/slurm_olmo_attention.sh`
   - Run analysis on alien partition
   - Generate outputs for all layers

3. **New Plot Script**: `experiments/compare_olmo_pythia_attention.py`
   - Load Pythia alluvial data
   - Load OLMo analysis data
   - Create side-by-side comparison

## Expected Outputs

```
outputs/OLMo_attention/
└── allenai/
    └── OLMo-1B-hf/
        ├── layer_0/
        │   └── full_analysis_cyc0_ml32.out
        ├── layer_1/
        ...
        └── layer_15/
            └── full_analysis_cyc0_ml32.out
```

## Key Questions to Answer

1. ✅ Can we adapt `aa_fortu.py` to OLMo? **YES - needs hook path changes only**
2. ⚠️ Do we have OLMo checkpoints? **NO - need alternative approach**
3. ✅ Can we generate meaningful comparison? **YES - with single-model or family comparison**

## Next Steps

1. **Inspect OLMo model structure** to find correct hook paths
2. **Create adapted analysis script** with OLMo-specific settings
3. **Run test on 1-2 layers** to validate approach
4. **Scale to all layers** once validated
5. **Generate comparison visualizations**
