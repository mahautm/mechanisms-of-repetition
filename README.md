# Parrots: Repetition Analysis in Language Models

This repository contains code for analyzing repetitive text generation behavior in transformer language models, with a focus on slot-filling tasks and attention patterns.

## 📄 Paper

The main paper is available in this repository: **[`repetition_alluvial_paper.pdf`](repetition_alluvial_paper.pdf)**

The paper investigates how repetition behavior emerges during model training using Pythia models at various checkpoints.

## 📚 Documentation

**→ [Full Documentation Index](docs/README.md)** - Start here for comprehensive guides

### Quick Links

| Document | Purpose | Best For |
|----------|---------|----------|
| **[Quick Start](docs/03-QUICK-START.md)** | Commands and workflows | Running experiments now |
| **[Experiments Overview](docs/01-EXPERIMENTS-OVERVIEW.md)** | What's in this repo | Understanding the code |
| **[Using OLMo](docs/02-USING-OLMO.md)** | OLMo model adaptation | Switching from Pythia to OLMo |
| **[Cycle Analysis](cycle-attention-analysis/README.md)** | Repetition mechanisms | Understanding repetition causality |

## 🚀 Quick Start

### Installation

```bash
# Install dependencies with Poetry
poetry install

# Test installation
python -c "from parrots.archs import get_model; model, tok = get_model('EleutherAI/pythia-1.4b'); print('✓ OK')"
```

### Run Paper Experiments

```bash
# Slot-filling evaluation (Pythia)
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    EleutherAI/pythia-1.4b \
    outputs/pythia-test/results.csv

# Generate alluvial plots (requires pre-computed data)
python run_alluvial_dual.py
python run_cycle_evolution.py
```

### Run Causal Head Ablation Experiments (Head-by-Head, Cycle-by-Cycle)

```bash
# Smoke run (small, fast)
bash scripts/run_ablation_head_cycle_smoke.sh EleutherAI/pythia-70m 4 "step1 steplatest"

# Full sweep (all checkpoints, all heads in target layer)
bash scripts/run_ablation_head_cycle_full.sh EleutherAI/pythia-1.4b 19

# Analyze output deltas and rank heads by causal effect
srun --partition=alien --qos=alien --exclude=node044 python analyze_ablation_head_cycle.py \
    --overall_delta_csv outputs_ablation_head_cycle/head_cycle_ablation_EleutherAI_pythia-1.4b_L19_overall_delta.csv \
    --cycle_delta_csv outputs_ablation_head_cycle/head_cycle_ablation_EleutherAI_pythia-1.4b_L19_cycle_delta.csv \
    --output_dir outputs_ablation_head_cycle
```

### Try with OLMo

```bash
# Slot-filling evaluation (OLMo)
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    allenai/OLMo-1B-hf \
    outputs/olmo-test/results.csv
```

See [OLMO_ADAPTATION_GUIDE.md](OLMO_ADAPTATION_GUIDE.md) for complete instructions.

## 🧪 Experiment Organization

### Paper Experiments
The core experiments for the repetition alluvial paper:
- **Slot-filling evaluation** (`parrots/slot_filling.py`)
- **Cycle evolution analysis** (`run_cycle_evolution.py`)
- **Multi-head attention analysis** (`parrots/aa_fortu/`)
- **Alluvial visualizations** (`run_alluvial_dual.py`)

### Common Infrastructure
Shared utilities used across all experiments:
- Model loading (`parrots/archs.py`)
- Cycle detection (`parrots/cycle_detection.py`)
- Lens training (`parrots/aa_fortu/aa_fortu_train_*.py`)
- Evaluation metrics (`parrots/nli.py`)

### Other Experiments
Additional research not in the main paper:
- Entropy analysis (`logit_entropy_analysis.py`)
- Ablation studies (`ablation_results_*/`)
- Perturbation experiments (`parrots/perturbation_*.py`)
- Cycle-attention breakthrough research (`cycle-attention-analysis/`)

## 🔬 Supported Models

### Tested with Paper
- **EleutherAI/pythia-1.4b** (primary model)
- Checkpoints: step1, step1000, step5000, step10000, step100000, steplatest

### Compatible Models
The infrastructure supports any HuggingFace causal language model:
- **Pythia family:** 70m, 1.4b, 6.9b, 12b
- **OLMo family:** OLMo-1B-hf, OLMo-7B-hf, OLMo-2-*
- **Llama family:** Llama-3.2-1B, Llama-3.2-3B
- **Mistral, Qwen, Gemma, and others**

See [OLMO_ADAPTATION_GUIDE.md](OLMO_ADAPTATION_GUIDE.md) for model-specific instructions.

## 📊 Key Findings

From the paper:
- Repetition behavior emerges progressively during training
- Specific attention heads become specialized for cycle continuation
- Layer 19 shows critical repetition-related patterns
- ICL (In-Context Learning) conditions show different repetition dynamics than natural prompts

From cycle-attention-analysis:
- Semantic repetition patterns causally trigger repetitive generation
- "more and more and" → 288 repetitions (993.3x baseline effectiveness)
- Direct attention interventions are ineffective
- Newline tokens are correlated but not causal

## 📁 Repository Structure

```
parrots/
├── EXPERIMENT_ORGANIZATION.md    # Complete experiment guide
├── OLMO_ADAPTATION_GUIDE.md      # OLMo-specific instructions  
├── QUICK_REFERENCE.md            # Quick commands
├── repetition_alluvial_paper.pdf # The paper
├── pyproject.toml                # Dependencies
│
├── run_alluvial_dual.py          # Main visualization script
├── run_cycle_evolution.py        # Cycle evolution plots
├── run_mlp_evolution.py          # MLP evolution analysis
│
├── data/                         # Datasets
│   ├── human_lama_parrots_list_v1.csv
│   └── lama.csv
│
├── parrots/                      # Main package
│   ├── archs.py                  # Model utilities
│   ├── slot_filling.py           # Evaluation script
│   ├── cycle_detection.py        # Cycle detection
│   └── aa_fortu/                 # Attention analysis
│       ├── aa_fortu.py           # Main analysis
│       ├── multihead_analysis_graphs.py  # Visualizations
│       └── aa_fortu_train_*.py   # Lens training
│
├── scripts/                      # Batch scripts
│   └── run_full_multihead_analysis.sh
│
├── cycle-attention-analysis/     # Breakthrough research
│   └── README.md                 # Full documentation
│
└── outputs/                      # Results (generated)
    └── outputs_multihead_full/   # Multi-head analysis results
```

## 🔄 Replicating with OLMo

**Can most experiments be repeated with OLMo?** 

**Yes, with adaptations:**
- ✅ Slot-filling evaluation works directly
- ✅ Cycle detection works directly
- ✅ Attention analysis works with layer count adjustment
- ⚠️ Training evolution requires OLMo checkpoints (may not be available)
- ⚠️ Architecture differences require parameter adjustments

**Recommended approach:**
1. Start with final OLMo model analysis
2. Compare OLMo vs Pythia behavior
3. If checkpoints available, replicate training evolution
4. Focus on novel OLMo-specific insights

See [OLMO_ADAPTATION_GUIDE.md](OLMO_ADAPTATION_GUIDE.md) for complete instructions, including:
- Architecture difference handling
- Checkpoint availability and workarounds
- Complete adaptation scripts
- Comparison study templates

## 🛠️ Common Tasks

### Run Slot-Filling with Any Model
```bash
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    your-model-name \
    outputs/your-model/results.csv
```

### Analyze Specific Checkpoint
```bash
python -m parrots.slot_filling \
    data/human_lama_parrots_list_v1.csv \
    EleutherAI/pythia-1.4b \
    outputs/pythia-step1000/results.csv \
    --revision step1000
```

### Compare Multiple Models
```bash
for model in "EleutherAI/pythia-1.4b" "allenai/OLMo-1B-hf"; do
    python -m parrots.slot_filling \
        data/human_lama_parrots_list_v1.csv \
        ${model} \
        outputs/$(echo ${model} | tr '/' '_')/results.csv
done
```

## 📖 Citation

If you use this code, please cite the repetition alluvial paper (see [`repetition_alluvial_paper.pdf`](repetition_alluvial_paper.pdf)).

## 🤝 Getting Help

**[→ Documentation Index](docs/README.md)** - All guides in one place

For specific questions:
- **Running experiments:** [Quick Start Guide](docs/03-QUICK-START.md)
- **Understanding code:** [Experiments Overview](docs/01-EXPERIMENTS-OVERVIEW.md)
- **Using OLMo:** [OLMo Adaptation Guide](docs/02-USING-OLMO.md)
- **Repetition mechanisms:** [Cycle Analysis Research](cycle-attention-analysis/README.md)

## 📝 License

[Specify license here]

## 👥 Authors

Mateo Mahaut (mateo.mahaut@gmail.com)
