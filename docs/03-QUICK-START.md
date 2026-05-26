# Quick Reference: Running Experiments

This is a quick reference guide for common tasks. See `EXPERIMENT_ORGANIZATION.md` for detailed documentation.

## Quick Start

### 1. Test Installation
```bash
# Verify dependencies
poetry install

# Test model loading
python -c "from parrots.archs import get_model; model, tok = get_model('EleutherAI/pythia-1.4b'); print('✓ OK')"
```

### 2. Run Paper Experiments (Pythia)

#### Slot-Filling Evaluation
```bash
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    EleutherAI/pythia-1.4b \
    outputs/pythia-test/results.csv \
    --batch-size 8 \
    --max-new-tokens 20
```

#### Generate Alluvial Plots
```bash
# Requires pre-computed multihead analysis data
python run_alluvial_dual.py
python run_cycle_evolution.py
python run_mlp_evolution.py
```

#### Multi-head Analysis (SLURM)
```bash
# Edit script to configure model/checkpoints
bash scripts/run_full_multihead_analysis.sh
```

### 3. Run Experiments with OLMo

#### Quick Test
```bash
# Test OLMo-1B
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    allenai/OLMo-1B-hf \
    outputs/olmo-test/results.csv \
    --batch-size 8
```

#### Full OLMo Analysis
```bash
# See OLMO_ADAPTATION_GUIDE.md for complete instructions

# 1. Edit model name in scripts
MODEL="allenai/OLMo-1B-hf"
MAX_LAYERS=16  # OLMo-1B has 16 layers

# 2. Run analysis
bash run_olmo_multihead_analysis.sh  # Create this from template in guide
```

## Common Tasks

### Change Model
Replace model name in any script:
```python
# From:
model_name = "EleutherAI/pythia-1.4b"

# To:
model_name = "allenai/OLMo-1B-hf"
# OR
model_name = "meta-llama/Llama-3.2-1B"
# OR
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
```

### Adjust for Different Architecture
```python
# Check model layers
from transformers import AutoConfig
config = AutoConfig.from_pretrained("your-model-name")
print(f"Layers: {config.num_hidden_layers}")

# Update max_layer_idx accordingly
max_layer_idx = config.num_hidden_layers
```

### Use Specific Checkpoint
```python
# Pythia checkpoints
revision = "step1000"  # or step1, step5000, step10000, step100000

# Load with checkpoint
model, tok = get_model("EleutherAI/pythia-1.4b", revision=revision)
```

## File Locations

### Input Data
- LAMA dataset: `data/human_lama_parrots_list_v1.csv`
- Autoprompts: `data/autoprompts_opt1_3b_lama_parrot_list_v1.csv`

### Outputs
- Slot-filling: `outputs/{model_name}/slot_filling_results.csv`
- Perturbations: `outputs/{model_name}/perturbations/`
- Multi-head analysis: `outputs_multihead_full/{model_name}/`

### Visualizations
- Paper plots: Root directory (e.g., `alluvial_layer_19_dual.png`)
- Cycle analysis plots: `cycle-attention-analysis/plots/`
- General plots: `plots/`

### Logs
- SLURM logs: `logs/`
- Analysis logs: Specified in `--log-file` parameter

## Troubleshooting

### Out of Memory
```python
# Use bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16
)

# OR use 8-bit quantization
from transformers import BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
```

### Model Not Found
```bash
# Check HuggingFace Hub
huggingface-cli login
huggingface-cli repo info allenai/OLMo-1B-hf

# Some models require authentication
# Get token from https://huggingface.co/settings/tokens
```

### Tokenizer Issues
```python
# Fix missing pad token
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## Key Scripts Reference

### Paper Experiments
- `run_alluvial_dual.py` - Dual alluvial plots (ICL vs Natural)
- `run_cycle_evolution.py` - Cycle evolution across checkpoints
- `run_mlp_evolution.py` - MLP layer evolution
- `parrots/slot_filling.py` - Slot-filling evaluation

### Analysis
- `parrots/aa_fortu/aa_fortu.py` - Attention analysis
- `parrots/cycle_detection.py` - Cycle detection
- `parrots/cycle_perturbation.py` - Perturbation experiments

### Visualization
- `parrots/aa_fortu/multihead_analysis_graphs.py` - Multi-head visualizations
- `logit_entropy_analysis.py` - Entropy analysis
- `replot_entropy_no_icl.py` - Entropy plots

### Utilities
- `parrots/archs.py` - Model loading
- `parrots/nli.py` - NLI checking
- `parrots/distance.py` - Distance metrics

## Model Support Matrix

| Model | Size | Tested | Checkpoints | Notes |
|-------|------|--------|-------------|-------|
| Pythia-1.4b | 1.4B | ✅ | ✅ step1-steplatest | Paper model |
| Pythia-6.9b | 6.9B | ✅ | ✅ step1-steplatest | Larger variant |
| OLMo-1B | 1.2B | ⚠️ | ❌ main only | See adaptation guide |
| OLMo-7B | 6.9B | ⚠️ | ❌ main only | Similar to Pythia-6.9b |
| Llama-3.2-1B | 1B | ⚠️ | ❌ final only | Compatible |
| Qwen2.5-1.5B | 1.5B | ⚠️ | ❌ final only | Compatible |

**Legend:**
- ✅ Fully tested and documented
- ⚠️ Compatible but requires adaptation
- ❌ Feature not available

## Documentation Files

- **`EXPERIMENT_ORGANIZATION.md`** - Complete experiment organization and paper details
- **`OLMO_ADAPTATION_GUIDE.md`** - Step-by-step guide for using OLMo models
- **`QUICK_REFERENCE.md`** - This file (quick commands and tips)
- **`cycle-attention-analysis/README.md`** - Breakthrough repetition research
- **`repetition_alluvial_paper.pdf`** - The paper

## Getting Help

1. Check documentation files above
2. Look at script help: `python script.py --help`
3. Check logs in `logs/` directory
4. Review error messages in SLURM `.err` files

## Example Workflows

### Workflow 1: Evaluate New Model on Slot-Filling
```bash
# 1. Set model name
MODEL="your-model-name"

# 2. Run evaluation
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    ${MODEL} \
    outputs/${MODEL}/results.csv

# 3. Check results
python -c "import pandas as pd; df = pd.read_csv('outputs/${MODEL}/results.csv'); print(df.describe())"
```

### Workflow 2: Compare Two Models
```bash
# Run both
for MODEL in "EleutherAI/pythia-1.4b" "allenai/OLMo-1B-hf"; do
    echo "Testing ${MODEL}..."
    python -m parrots.slot_filling \
        data/human_lama_parrots_list_v1.csv \
        ${MODEL} \
        outputs/$(echo ${MODEL} | tr '/' '_')/results.csv
done

# Compare (create custom comparison script)
python compare_results.py
```

### Workflow 3: Full Checkpoint Analysis (Pythia Only)
```bash
# 1. Generate perturbation data
python -m parrots.cycle_perturbation \
    --model-name EleutherAI/pythia-1.4b \
    --checkpoints step1 step1000 step10000 step100000 steplatest

# 2. Run multi-head analysis
bash scripts/run_full_multihead_analysis.sh

# 3. Generate visualizations
python run_alluvial_dual.py
python run_cycle_evolution.py
```

---

**For detailed information, see:**
- Full experiment details: `EXPERIMENT_ORGANIZATION.md`
- OLMo-specific instructions: `OLMO_ADAPTATION_GUIDE.md`
- Repetition research: `cycle-attention-analysis/README.md`
