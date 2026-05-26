# Project Structure Overview

Visual guide to the repository organization.

```
parrots/
│
├── 📄 README.md                          ← Start here! Project overview
├── 📄 repetition_alluvial_paper.pdf      ← The paper
├── 📦 pyproject.toml                     ← Dependencies
│
├── 📚 docs/                              ← Documentation hub
│   ├── README.md                         ← Documentation index
│   ├── 01-EXPERIMENTS-OVERVIEW.md        ← What's in this repo
│   ├── 02-USING-OLMO.md                  ← OLMo adaptation guide
│   └── 03-QUICK-START.md                 ← Quick commands
│
├── 🎨 Paper Experiment Scripts           ← Main paper results
│   ├── run_alluvial_dual.py              ← Dual alluvial plots (main figure)
│   ├── run_cycle_evolution.py            ← Cycle evolution across training
│   └── run_mlp_evolution.py              ← MLP layer evolution
│
├── 📊 data/                              ← Input datasets
│   ├── human_lama_parrots_list_v1.csv    ← Slot-filling dataset
│   ├── lama.csv                          ← Original LAMA
│   └── autoprompts_*.csv                 ← Generated prompts
│
├── 🐍 parrots/                           ← Main Python package
│   │
│   ├── 🔧 Core Utilities
│   │   ├── archs.py                      ← Model loading (ANY model)
│   │   ├── cycle_detection.py            ← Cycle detection
│   │   └── nli.py                        ← NLI evaluation
│   │
│   ├── 🎯 Main Experiments
│   │   ├── slot_filling.py               ← Slot-filling evaluation
│   │   ├── cycle_perturbation.py         ← Perturbation experiments
│   │   └── perturbation_analysis*.py     ← Analysis scripts
│   │
│   └── 🔬 aa_fortu/                      ← Attention analysis (paper core)
│       ├── aa_fortu.py                   ← Main attention analysis
│       ├── multihead_analysis_graphs.py  ← Visualization
│       ├── aa_fortu_train_lens.py        ← Linear probes
│       ├── aa_fortu_train_mlp_lens.py    ← MLP probes
│       └── aa_fortu_train_multihead_lens.py ← Multi-head probes
│
├── 🔬 cycle-attention-analysis/          ← Breakthrough research
│   ├── README.md                         ← Research overview
│   ├── docs/BREAKTHROUGH_SUMMARY.md      ← Key findings
│   ├── experiments/                      ← 6 phases of experiments
│   ├── reports/                          ← Detailed reports
│   └── src/                              ← Analysis utilities
│
├── 🚀 scripts/                           ← Batch processing scripts
│   ├── run_full_multihead_analysis.sh    ← SLURM batch analysis
│   ├── train_mlp_lenses.sh               ← Lens training
│   ├── slurm_sf.sh                       ← Slot-filling batch
│   └── perturbated.sh                    ← Perturbation batch
│
├── 📈 Outputs (generated during runs)
│   ├── outputs/                          ← Slot-filling results
│   ├── outputs_multihead_full/           ← Multi-head analysis
│   ├── lenses/                           ← Trained linear probes
│   ├── lenses_mlp/                       ← Trained MLP probes
│   ├── lenses_multihead/                 ← Multi-head probes
│   ├── plots/                            ← Generated plots
│   └── logs/                             ← Execution logs
│
└── 🧪 Other Experiments
    ├── logit_entropy_analysis.py         ← Entropy analysis
    ├── entropy_analysis.py               
    ├── replot_entropy_no_icl.py          
    └── ablation_results_*/               ← Ablation studies
```

## 🎯 Key Paths for Common Tasks

### Running Paper Experiments
```bash
# Main visualization scripts (root directory)
python run_alluvial_dual.py
python run_cycle_evolution.py
python run_mlp_evolution.py

# Core evaluation
python -m parrots.slot_filling [args]

# Batch analysis (SLURM)
bash scripts/run_full_multihead_analysis.sh
```

### Input Data Locations
```
data/human_lama_parrots_list_v1.csv    # Primary dataset
data/lama.csv                           # Original LAMA
```

### Output Data Locations
```
outputs/{model_name}/                   # Model-specific outputs
outputs_multihead_full/                 # Multi-head analysis
plots/                                  # Generated visualizations
logs/                                   # Execution logs
```

### Documentation Locations
```
docs/README.md                          # Documentation hub
docs/01-EXPERIMENTS-OVERVIEW.md         # Complete experiment guide
docs/02-USING-OLMO.md                   # OLMo adaptation
docs/03-QUICK-START.md                  # Quick reference
cycle-attention-analysis/README.md      # Repetition research
```

## 🔍 Find Files by Purpose

### Core Infrastructure (use with any model)
- Model loading: `parrots/archs.py`
- Cycle detection: `parrots/cycle_detection.py`
- Evaluation: `parrots/nli.py`, `parrots/slot_filling.py`

### Paper Core (attention analysis)
- Main analysis: `parrots/aa_fortu/aa_fortu.py`
- Visualizations: `parrots/aa_fortu/multihead_analysis_graphs.py`
- Training evolution: `run_cycle_evolution.py`, `run_mlp_evolution.py`
- Main figure: `run_alluvial_dual.py`

### Lens Training (probing)
- Linear: `parrots/aa_fortu/aa_fortu_train_lens.py`
- MLP: `parrots/aa_fortu/aa_fortu_train_mlp_lens.py`
- Multi-head: `parrots/aa_fortu/aa_fortu_train_multihead_lens.py`
- Batch script: `scripts/train_mlp_lenses.sh`

### Batch Processing
- Multi-head analysis: `scripts/run_full_multihead_analysis.sh`
- Slot-filling: `scripts/slurm_sf.sh`
- Perturbations: `scripts/perturbated.sh`

### Exploratory Experiments
- Entropy: `logit_entropy_analysis.py`, `replot_entropy_no_icl.py`
- Ablations: `ablation_results_*/`
- Distance analysis: `parrots/distance*.py`
- 2D projections: `parrots/2d_mapping*.py`

### Repetition Research
- Main research: `cycle-attention-analysis/`
- Experiments: `cycle-attention-analysis/experiments/`
- Findings: `cycle-attention-analysis/docs/BREAKTHROUGH_SUMMARY.md`

## 📝 Configuration Files

```
pyproject.toml                          # Python dependencies (Poetry)
poetry.lock                             # Locked dependency versions
```

## 🗂️ Data Organization Pattern

All model outputs follow this pattern:
```
outputs/
└── {model_org}/
    └── {model_name}_human_lama_parrots_list_v1_sf/
        ├── slot_filling_results.csv
        └── perturbations/
            ├── cycle_3_results_*.csv
            ├── cycle_4_results_*.csv
            └── cycle_5_results_*.csv
```

Multi-head analysis results:
```
outputs_multihead_full/
└── {model_org}/
    └── {model_name}/
        ├── step1/
        │   └── layer_{0..N}/
        │       └── full_analysis_cyc{0..5}_ml32.out
        ├── step1000/
        ├── step5000/
        ├── step10000/
        ├── step100000/
        └── steplatest/
```

## 🎨 Visualization Outputs

Paper figures (root directory):
```
alluvial_layer_19_dual.{png,pdf}        # Main dual alluvial
cycle_evolution_horizontal.png          # Cycle evolution
mlp_evolution_no_step7000.png          # MLP evolution
logit_entropy_evolution_no_icl.png     # Entropy evolution
```

Additional plots:
```
plots/                                  # Various analysis plots
cycle-attention-analysis/plots/         # Repetition research plots
outputs_multihead_full/alluvial_plots_multi_category/  # Category plots
```

## 🔢 Model Architecture Quick Reference

### Pythia-1.4b (paper model)
- Layers: 24 (0-23)
- Heads: 16 per layer
- Hidden size: 2048
- Vocab: 50304

### OLMo-1B (comparable)
- Layers: 16 (0-15)  ⚠️ Different!
- Heads: 16 per layer
- Hidden size: 2048
- Vocab: 50280

### Adjusting for Different Models
```python
# Check any model's architecture
from transformers import AutoConfig
config = AutoConfig.from_pretrained("model-name")
print(f"Layers: {config.num_hidden_layers}")
print(f"Heads: {config.num_attention_heads}")
```

## 🚦 Entry Points

### For Users
1. **Quick start:** [docs/03-QUICK-START.md](03-QUICK-START.md)
2. **Run paper experiments:** `run_alluvial_dual.py`
3. **Evaluate model:** `python -m parrots.slot_filling`

### For Developers
1. **Understand structure:** [docs/01-EXPERIMENTS-OVERVIEW.md](01-EXPERIMENTS-OVERVIEW.md)
2. **Adapt for new model:** [docs/02-USING-OLMO.md](02-USING-OLMO.md)
3. **Core utilities:** `parrots/archs.py`, `parrots/cycle_detection.py`

### For Researchers
1. **Paper experiments:** `parrots/aa_fortu/`
2. **Repetition mechanisms:** `cycle-attention-analysis/`
3. **Training evolution:** `run_cycle_evolution.py`, `run_mlp_evolution.py`

---

**Need more detail?** See the full documentation in the [`docs/`](README.md) directory.
