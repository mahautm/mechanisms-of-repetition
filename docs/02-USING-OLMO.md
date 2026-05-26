# OLMo Adaptation Guide

This guide provides detailed instructions for adapting the experiments from the paper to use OLMo models instead of Pythia.

## Table of Contents
1. [OLMo Model Variants](#olmo-model-variants)
2. [Architecture Differences](#architecture-differences)
3. [Step-by-Step Adaptation](#step-by-step-adaptation)
4. [Checkpoint Availability](#checkpoint-availability)
5. [Known Issues and Solutions](#known-issues-and-solutions)
6. [Comparison Study Template](#comparison-study-template)

---

## OLMo Model Variants

### Available OLMo Models on HuggingFace

#### OLMo 1.0 Family
- **allenai/OLMo-1B-hf** (1.2B params)
- **allenai/OLMo-7B-hf** (6.9B params)
- **allenai/OLMo-1.7-7B-hf** (7B params)

#### OLMo 2.0 Family (Latest)
- **allenai/OLMo-2-1124-7B** (7B params, Nov 2024)
- **allenai/OLMo-2-1124-13B** (13B params, Nov 2024)
- **allenai/OLMo-2-0425-1B-Instruct** (1B params, instruction-tuned)

### Recommended for Paper Replication

**Best match to Pythia-1.4b:** `allenai/OLMo-1B-hf`
- Similar size (1.2B vs 1.4B parameters)
- Base (non-instruct) model for fair comparison
- Standard architecture without instruction-tuning bias

**Alternative:** `allenai/OLMo-7B-hf`
- Comparable to Pythia-6.9b
- More capacity, may show different repetition patterns

---

## Architecture Differences

### Pythia-1.4b Architecture
```python
{
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "vocab_size": 50304,
    "max_position_embeddings": 2048
}
```

### OLMo-1B Architecture
```python
{
    "hidden_size": 2048,
    "num_hidden_layers": 16,  # DIFFERENT: 16 vs 24
    "num_attention_heads": 16,
    "vocab_size": 50280,  # DIFFERENT: slightly smaller
    "max_position_embeddings": 2048
}
```

### Key Differences to Handle

1. **Layer Count:**
   - Pythia-1.4b: 24 layers
   - OLMo-1B: 16 layers
   - **Action:** Adjust `max_layer_idx` parameters

2. **Vocabulary Size:**
   - Pythia: 50304 tokens
   - OLMo: 50280 tokens
   - **Action:** Mostly handled automatically by tokenizer

3. **Tokenizer:**
   - Both use GPT-2 style BPE
   - May have slight differences in special tokens
   - **Action:** Check pad_token, eos_token configuration

4. **Architecture Details:**
   - OLMo uses different normalization (RMSNorm)
   - Different initialization schemes
   - **Action:** No code changes needed (handled by transformers)

---

## Step-by-Step Adaptation

### Phase 1: Basic Compatibility Testing

#### Test 1: Model Loading
```python
# Test basic model loading
python -c "
from parrots.archs import get_model, get_tokenizer
import torch

model_name = 'allenai/OLMo-1B-hf'
print(f'Loading {model_name}...')

model, tokenizer = get_model(model_name)
print(f'Model loaded: {model.config.num_hidden_layers} layers')
print(f'Vocab size: {model.config.vocab_size}')
print(f'Hidden size: {model.config.hidden_size}')
print(f'Attention heads: {model.config.num_attention_heads}')

# Test generation
text = 'The capital of France is'
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
print(f'Generated: {tokenizer.decode(outputs[0])}')
print('✓ Model loading successful!')
"
```

#### Test 2: Cycle Detection
```python
# Test cycle detection on OLMo
python -c "
from parrots.archs import get_model
from parrots.cycle_detection import detect_cycles
import torch

model, tokenizer = get_model('allenai/OLMo-1B-hf')

# Test with a prompt that might induce repetition
prompt = 'The cat sat on the cat sat on the'
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50,
        do_sample=False  # Greedy decoding
    )

generated = tokenizer.decode(outputs[0])
print(f'Generated text: {generated}')

# Detect cycles (you may need to adjust import based on actual function)
# This is a placeholder - adjust to your actual cycle detection code
print('✓ Generation successful, check for cycles manually')
"
```

#### Test 3: Slot-Filling Evaluation
```bash
# Run slot-filling on small subset
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    allenai/OLMo-1B-hf \
    outputs/OLMo-1B-test/slot_filling_results.csv \
    --batch-size 4 \
    --max-new-tokens 20 \
    --log-file logs/olmo_test.log

# Check outputs
head outputs/OLMo-1B-test/slot_filling_results.csv
```

### Phase 2: Attention Analysis Adaptation

#### Step 1: Create OLMo-specific Analysis Script

Create `run_olmo_multihead_analysis.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=olmo_multihead
#SBATCH --output=logs/olmo_%A_%a.out
#SBATCH --error=logs/olmo_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# OLMo-1B Configuration
MODEL_NAME="allenai/OLMo-1B-hf"
OUTPUT_DIR="/home/mmahaut/projects/parrots/outputs_multihead_full"
LENS_PATH="/home/mmahaut/projects/parrots/lenses_multihead"
BASE_PATH="/home/mmahaut/projects/parrots/outputs/${MODEL_NAME}_human_lama_parrots_list_v1_sf/perturbations"

# OLMo-1B has 16 layers (0-15)
MAX_LAYER=15

# NOTE: OLMo doesn't have intermediate checkpoints by default
# Use "main" for the final checkpoint, or specific commit hashes if available
checkpoint_list=("main")

# Cycle configurations
cycle_list=(0 1 2 3)
max_length=32

for checkpoint in "${checkpoint_list[@]}"; do
    echo "Processing checkpoint: ${checkpoint}"
    
    # Run analysis for each layer
    for layer_idx in $(seq 0 ${MAX_LAYER}); do
        echo "  Analyzing layer ${layer_idx}..."
        
        JOB_NAME="olmo_mh_L${layer_idx}"
        SPECIFIC_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}/${checkpoint}/layer_${layer_idx}"
        
        mkdir -p "${SPECIFIC_OUTPUT}"
        
        # Run the multi-head analysis
        python -m parrots.aa_fortu.aa_fortu_train_multihead_lens \
            --model-name="${MODEL_NAME}" \
            $([ "${checkpoint}" != "main" ] && echo "--revision=${checkpoint}") \
            --base-path="${BASE_PATH}" \
            --max-layer-idx=$((MAX_LAYER + 1)) \
            --single-layer="${layer_idx}" \
            --lens-path="${LENS_PATH}" \
            --output-path="${SPECIFIC_OUTPUT}" \
            --n-cycles="${cycle_list[@]}" \
            --max-length="${max_length}" \
            --use-bfloat16
        
        echo "  ✓ Layer ${layer_idx} complete"
    done
done

echo "✓ OLMo multi-head analysis complete!"
```

#### Step 2: Adjust Python Scripts for OLMo

Key modifications needed in analysis scripts:

```python
# In run_cycle_evolution.py or similar scripts

# BEFORE (Pythia):
results_across_cycles = load_multihead_results_across_cycles(
    base_path='/home/mmahaut/projects/parrots/outputs_multihead_full',
    model_name='EleutherAI/pythia-1.4b',
    checkpoints=['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'steplatest'],
    cycle_range=[0, 1, 2, 3, 4, 5],
    max_length=32
)

# AFTER (OLMo):
results_across_cycles = load_multihead_results_across_cycles(
    base_path='/home/mmahaut/projects/parrots/outputs_multihead_full',
    model_name='allenai/OLMo-1B-hf',
    checkpoints=['main'],  # Or specific checkpoints if available
    cycle_range=[0, 1, 2, 3, 4, 5],
    max_length=32
)
```

### Phase 3: Full Experiment Pipeline

#### Complete OLMo Experiment Workflow

Create `run_olmo_full_experiment.sh`:

```bash
#!/bin/bash
# Complete OLMo experiment pipeline

MODEL="allenai/OLMo-1B-hf"
SAFE_NAME="allenai_OLMo-1B-hf"
BASE_DIR="/home/mmahaut/projects/parrots"

echo "=== OLMo Full Experiment Pipeline ==="
echo "Model: ${MODEL}"
echo "Start time: $(date)"

# Step 1: Slot-filling evaluation
echo -e "\n[1/5] Running slot-filling evaluation..."
python -m parrots.slot_filling \
    ${BASE_DIR}/data/human_lama_parrots_list_v1.csv \
    ${MODEL} \
    ${BASE_DIR}/outputs/${SAFE_NAME}/slot_filling_results.csv \
    --batch-size 8 \
    --max-new-tokens 20 \
    --log-file ${BASE_DIR}/logs/olmo_slot_filling.log

# Step 2: Cycle perturbation (generate data)
echo -e "\n[2/5] Running cycle perturbation analysis..."
python -m parrots.cycle_perturbation \
    --model-name ${MODEL} \
    --data-path ${BASE_DIR}/data/human_lama_parrots_list_v1.csv \
    --output-path ${BASE_DIR}/outputs/${SAFE_NAME}/perturbations \
    --cycle-sizes 3 4 5

# Step 3: Attention analysis
echo -e "\n[3/5] Running attention analysis..."
python -m parrots.aa_fortu.aa_fortu \
    --model-name ${MODEL} \
    --base-path ${BASE_DIR}/outputs/${SAFE_NAME}/perturbations \
    --max-layer-idx 16 \
    --n-cycles 1 \
    --use-bfloat16

# Step 4: Multi-head analysis (if needed)
echo -e "\n[4/5] Running multi-head analysis..."
bash run_olmo_multihead_analysis.sh

# Step 5: Visualization
echo -e "\n[5/5] Generating visualizations..."
# Note: May need to adapt visualization scripts for single checkpoint
python -c "
from parrots.aa_fortu.multihead_analysis_graphs import load_multihead_results_across_cycles
import matplotlib.pyplot as plt

# Load OLMo results
results = load_multihead_results_across_cycles(
    base_path='${BASE_DIR}/outputs_multihead_full',
    model_name='${MODEL}',
    checkpoints=['main'],
    cycle_range=[0, 1, 2, 3],
    max_length=32
)

# Create visualization
# (Adapt as needed for single-checkpoint analysis)
print('Results loaded, ready for visualization')
"

echo -e "\n=== OLMo Experiment Complete ==="
echo "End time: $(date)"
echo "Results in: ${BASE_DIR}/outputs/${SAFE_NAME}/"
```

---

## Checkpoint Availability

### Finding OLMo Checkpoints

#### Option 1: Use Final Model Only
```python
model_name = "allenai/OLMo-1B-hf"  # Final checkpoint
```

**Limitations:**
- Cannot replicate training evolution analysis
- No alluvial plots across training stages
- Focus on final model behavior only

#### Option 2: Find Intermediate Checkpoints

AllenAI may provide intermediate checkpoints. Check:

1. **HuggingFace Hub:**
```bash
# List all OLMo models
huggingface-cli repo list-files allenai/OLMo-1B-hf

# Check for branches/tags with checkpoint info
git clone https://huggingface.co/allenai/OLMo-1B-hf
cd OLMo-1B-hf
git branch -a
git tag -l
```

2. **AllenAI AI2 Cloud:**
- Check AllenAI's documentation
- Intermediate checkpoints may be available on AI2's cloud storage
- May need to download and upload to HuggingFace

3. **Training from Scratch:**
- Use AllenAI's OLMo training code
- Train with checkpointing at specific intervals
- Most resource-intensive but gives full control

#### Option 3: Use Commit Hashes as "Checkpoints"

If model was updated over time:
```python
# Different versions as pseudo-checkpoints
checkpoints = [
    "commit_hash_1",  # Earlier version
    "commit_hash_2",  # Later version  
    "main"            # Latest
]
```

### Workaround: Cross-Sectional Instead of Longitudinal

If checkpoints unavailable, adapt the study:

**Original Paper:** Tracks repetition behavior across training (longitudinal)

**OLMo Adaptation:** Compare different OLMo variants (cross-sectional)
```python
# Compare different OLMo models instead of checkpoints
olmo_variants = [
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-7B-hf",
    "allenai/OLMo-2-1124-7B",
]

# Or compare OLMo to other models
models = [
    "allenai/OLMo-1B-hf",
    "EleutherAI/pythia-1.4b",
    "meta-llama/Llama-3.2-1B",
]
```

---

## Known Issues and Solutions

### Issue 1: Different Tokenization
**Problem:** OLMo may tokenize slot-filling prompts differently

**Solution:**
```python
# Add tokenizer comparison in evaluation
pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
olmo_tok = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")

# Compare tokenization
test_text = "The capital of France is"
print(f"Pythia: {pythia_tok.tokenize(test_text)}")
print(f"OLMo: {olmo_tok.tokenize(test_text)}")

# Document differences in results
```

### Issue 2: Fewer Layers
**Problem:** OLMo-1B has 16 layers vs Pythia's 24

**Solution:**
```python
# Option A: Analyze all OLMo layers (0-15)
# Compare to subset of Pythia layers

# Option B: Focus on specific layer indices
# E.g., if paper focuses on layer 19 (79% through model)
# For OLMo: use layer 12-13 (75-81% through model)
target_layer = int(0.79 * olmo_model.config.num_hidden_layers)
```

### Issue 3: No Training Evolution Data
**Problem:** Can't replicate alluvial plots without checkpoints

**Solutions:**

**Option A:** Focus on other aspects
- Attention pattern analysis
- Cycle detection mechanisms
- Repetition triggers

**Option B:** Create novel OLMo-specific analysis
- Compare OLMo-1B vs OLMo-7B
- OLMo base vs instruct models
- OLMo vs Pythia final models

**Option C:** Use model pruning as proxy for training stages
```python
# Progressively prune model to simulate earlier training
# Layer pruning, attention head pruning, etc.
# Analyze repetition behavior as model capacity decreases
```

### Issue 4: Memory Issues with Larger Models
**Problem:** OLMo-7B may exceed GPU memory

**Solution:**
```python
# Use 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-7B-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

## Comparison Study Template

### Recommended: OLMo vs Pythia Comparison

Instead of replicating the paper exactly, create a comparative study:

#### Research Questions:
1. Do OLMo and Pythia show similar repetition patterns?
2. Which attention heads are critical for repetition in each model?
3. Are slot-filling accuracies comparable?
4. Do cycle detection patterns differ?

#### Experiment Design:

```python
# experiments/olmo_pythia_comparison.py

models = {
    'pythia-1.4b': {
        'name': 'EleutherAI/pythia-1.4b',
        'checkpoint': 'steplatest',
        'layers': 24,
        'compare_layer': 19,  # From paper
    },
    'olmo-1b': {
        'name': 'allenai/OLMo-1B-hf',
        'checkpoint': 'main',
        'layers': 16,
        'compare_layer': 12,  # ~75% through model (similar to layer 19/24)
    }
}

def compare_models(models, experiments=['slot_filling', 'attention', 'cycles']):
    """Run comparison experiments"""
    
    results = {}
    
    for model_key, model_config in models.items():
        print(f"\n=== Analyzing {model_key} ===")
        
        results[model_key] = {}
        
        if 'slot_filling' in experiments:
            results[model_key]['slot_filling'] = run_slot_filling(
                model_config['name'], 
                model_config['checkpoint']
            )
        
        if 'attention' in experiments:
            results[model_key]['attention'] = analyze_attention_patterns(
                model_config['name'],
                model_config['compare_layer']
            )
        
        if 'cycles' in experiments:
            results[model_key]['cycles'] = detect_repetition_cycles(
                model_config['name']
            )
    
    return results

# Run comparison
results = compare_models(models)

# Visualize differences
create_comparison_plots(results, save_path='plots/olmo_vs_pythia/')
```

#### Comparison Metrics:

1. **Slot-Filling Performance:**
   - Direct follow accuracy
   - Exact match rate
   - NLI factual equivalence

2. **Repetition Behavior:**
   - Cycle detection frequency
   - Average cycle length
   - Trigger patterns

3. **Attention Patterns:**
   - Critical attention heads
   - Layer-wise importance
   - Head specialization

4. **Generate Comparative Visualizations:**
   - Side-by-side alluvial diagrams (if multiple checkpoints)
   - Attention heatmap comparisons
   - Performance metric tables

---

## Example: Complete OLMo Single-Model Analysis

For a simplified single-model analysis (no checkpoints):

```python
# scripts/analyze_olmo_final_model.py

"""
OLMo Final Model Analysis
Analyzes repetition behavior in final OLMo-1B checkpoint
"""

import torch
from pathlib import Path
from parrots.archs import get_model
from parrots.slot_filling import slot_fill
from parrots.cycle_detection import detect_cycles
from parrots.aa_fortu.aa_fortu import extract_contrasts

def main():
    model_name = "allenai/OLMo-1B-hf"
    output_dir = Path("outputs/OLMo-1B-final-analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {model_name}...")
    model, tokenizer = get_model(model_name)
    
    # 1. Slot-filling evaluation
    print("\n=== Slot-Filling Evaluation ===")
    sf_results = run_slot_filling_analysis(
        model, tokenizer, 
        data_path="data/human_lama_parrots_list_v1.csv",
        output_path=output_dir / "slot_filling.csv"
    )
    print(f"Accuracy: {sf_results['accuracy']:.2%}")
    
    # 2. Cycle detection
    print("\n=== Cycle Detection ===")
    cycle_results = run_cycle_detection(
        model, tokenizer,
        output_path=output_dir / "cycles.json"
    )
    print(f"Repetition rate: {cycle_results['repetition_rate']:.2%}")
    
    # 3. Layer-wise analysis
    print("\n=== Layer-wise Attention Analysis ===")
    for layer in [0, 4, 8, 12, 15]:  # Sample layers
        attention_results = analyze_layer_attention(
            model, tokenizer, layer,
            output_path=output_dir / f"layer_{layer}_attention.png"
        )
        print(f"Layer {layer}: {attention_results['top_heads']}")
    
    # 4. Generate report
    print("\n=== Generating Report ===")
    generate_analysis_report(
        sf_results, cycle_results, attention_results,
        output_path=output_dir / "analysis_report.md"
    )
    
    print(f"\n✓ Analysis complete! Results in {output_dir}")

if __name__ == "__main__":
    main()
```

---

## Summary

### ✅ Feasible with OLMo:
- Slot-filling evaluation
- Cycle detection
- Attention analysis
- Single-checkpoint full analysis
- Comparative studies with Pythia

### ⚠️ Challenging:
- Training evolution (requires intermediate checkpoints)
- Exact replication of alluvial plots
- Direct layer-to-layer mapping (different architectures)

### 💡 Recommended Approach:
1. Start with single-checkpoint analysis
2. Compare OLMo final model to Pythia final model
3. Document differences in architecture/behavior
4. If checkpoints available, replicate training evolution
5. Focus on novel insights from OLMo's unique characteristics

### 🎯 Unique OLMo Opportunities:
- Open training data enables tracing repetition to sources
- Compare different OLMo variants (1B, 7B, instruct)
- Analyze OLMo-specific architectural choices
- Study effect of different training regimes

