# Parrots Project: Experiment Organization

This document organizes the experiments in this repository into three categories:
1. **Paper Experiments** - Core experiments for the repetition alluvial paper
2. **Common Core** - Shared utilities and infrastructure used across experiments
3. **Other Experiments** - Exploratory and related investigations

## Overview

This project investigates repetitive text generation behavior in transformer language models, particularly focusing on **slot-filling tasks** and **attention patterns** that lead to cyclic/repetitive outputs. The main experiments use **Pythia models** (EleutherAI/pythia-1.4b primarily) with various training checkpoints to understand how repetition behavior emerges during training.

---

## 1. Paper Experiments (Repetition Alluvial Paper)

The paper `repetition_alluvial_paper.pdf` focuses on tracking how repetition behavior evolves across model training checkpoints.

### Main Paper Scripts

#### Core Analysis Scripts

1. **`run_alluvial_dual.py`** - Main visualization script
   - Creates dual alluvial plots showing repetition evolution
   - Compares ICL (In-Context Learning) vs Natural conditions
   - Uses layer 19 attention analysis
   - Outputs: `alluvial_layer_19_dual.{png,pdf}`

2. **`run_cycle_evolution.py`** - Cycle evolution tracking
   - Loads multi-head results across training cycles
   - Analyzes checkpoints: step1, step1000, step5000, step10000, step100000, steplatest
   - Creates horizontal cycle evolution plots
   - Outputs: `cycle_evolution_horizontal.png`

3. **`run_mlp_evolution.py`** - MLP layer evolution analysis
   - Tracks MLP layer behavior during training
   - Analyzes multiple checkpoints
   - Creates MLP evolution plots
   - Uses data from `test_mlp_pipeline_output/`

#### Data Generation Scripts

4. **`parrots/slot_filling.py`** - Slot-filling task evaluation
   - Main evaluation script for factual knowledge tasks
   - Uses LAMA dataset (`data/human_lama_parrots_list_v1.csv`)
   - Measures: direct_follow, exact_match, NLI factual equivalence
   - Generates outputs to `outputs/` directory structure

5. **`parrots/aa_fortu/aa_fortu.py`** - Attention analysis for repetition
   - Extracts attention head contrasts during cyclic generation
   - Uses transformer_lens for hooked model analysis
   - Analyzes which attention heads predict cycle continuation
   - Input: perturbation results from `outputs/.../perturbations/`
   - Supports lens transformations for layer analysis

6. **`parrots/aa_fortu/multihead_analysis_graphs.py`** - Multi-head visualization
   - Loads and visualizes multi-head attention results across cycles
   - Creates alluvial plots showing repetition emergence
   - Implements progressive categorization logic
   - Key functions:
     - `load_multihead_results_across_cycles()`
     - `plot_cycle_evolution_by_checkpoint()`
     - `create_progressive_categorization()`

### Supporting Scripts

7. **`parrots/cycle_perturbation.py`** - Cycle detection and perturbation
   - Detects repetitive cycles in generated text
   - Creates perturbation experiments
   - Generates data for attention analysis

8. **`parrots/aa_fortu/ckpt_pipeline_main.py`** - Checkpoint pipeline
   - Processes multiple model checkpoints
   - Orchestrates analysis across training stages

### Training Checkpoints Used

The paper analyzes Pythia-1.4b at these training stages:
- **step1** (143M tokens) - Initialization
- **step1000** (143B tokens) - Early training
- **step5000** (~715B tokens) - Mid training
- **step10000** (~1.43T tokens) - Mid-late training
- **step100000** (~14.3T tokens) - Late training  
- **steplatest** (Final) - Fully trained model

### Data Files

Key datasets used in paper experiments:
- `data/human_lama_parrots_list_v1.csv` - Human-curated LAMA slot-filling examples
- `data/lama.csv` - Original LAMA dataset
- `outputs_multihead_full/` - Full multi-head analysis results
- `test_mlp_pipeline_output/` - MLP evolution data

### Output Structure

Paper experiments generate results in:
```
outputs_multihead_full/
├── EleutherAI/pythia-1.4b/
│   ├── step1/layer_19/full_analysis_cyc0_ml32.out
│   ├── step1000/layer_19/...
│   ├── step5000/layer_19/...
│   ├── step10000/layer_19/...
│   ├── step100000/layer_19/...
│   └── steplatest/layer_19/...
└── alluvial_plots_multi_category/
```

### Paper Figures

Generated figures for the paper:
- `alluvial_layer_19_dual.{png,pdf}` - Main dual alluvial plot (ICL vs Natural) for Pythia-1.4b
- `cycle_evolution_horizontal.png` - Cycle evolution across checkpoints
- `mlp_evolution_no_step7000.png` - MLP layer evolution

### Additional Paper Experiments (Feb 2026)

These experiments extend the analysis to Pythia-70m and add new visualizations:

#### Entropy Evolution Analysis
Scripts: `entropy_evolution_analysis.py`, `scripts/run_entropy_evolution.sh`
- Tracks output entropy across training checkpoints
- Compares Natural (repetitive) vs No-Cycle-ICL (non-repetitive) sequences
- **Models**: Pythia-70m and Pythia-1.4b
- **Checkpoints**: step1, step1000, step5000, step10000, step100000, steplatest
- **Output**: `entropy_evolution_results/entropy_evolution_*.{json,png}`

#### Cycle Evolution: Original vs Learnt Repetition
Scripts: `cycle_evolution_original_vs_learnt.py`, `scripts/run_cycle_evolution_orig_learnt.sh`
- Distinguishes between:
  - **Original repetition**: Sequences repeating from step1 (inherent model bias)
  - **Learnt repetition**: Sequences that become repetitive during training
- Tracks when each sequence first starts repeating
- **Output**: `cycle_evolution_results/cycle_evolution_*.{json,png}`

#### Repetition Alluvial for Pythia-70m
Scripts: `create_repetition_alluvial.py`
- Creates alluvial visualization from cycle evolution data
- Shows how repetition composition changes across checkpoints
- Categories: Never repeats, Original, Learnt Early, Learnt Late
- **Output**: `alluvial_plots/repetition_alluvial_*.png`

#### Attention Fallback Analysis (Pythia-70m)
Location: `cycle-attention-analysis/experiments/phase3_newline_causality/`
- Analyzes attention redistribution when newlines are removed
- Compares Natural vs No-Cycle-ICL sequences
- **Key finding**: Pythia-70m has ~99.9% repetition rate (995/996 samples)
- **Output**: `plots/attention_fallback_alluvial_EleutherAI_pythia-70m_seed42/`

---

## 2. Common Core Infrastructure

Shared utilities and infrastructure used by all experiments.

### Model Loading and Utilities

1. **`parrots/archs.py`** - Model architecture utilities
   - `get_model()` - Load any HuggingFace causal LM
   - `get_tokenizer()` - Load and configure tokenizer
   - Supports all transformer models (Pythia, OLMo, Llama, etc.)
   - Handles padding token configuration

2. **`parrots/aa_fortu/modules/`** - Analysis modules
   - Model utilities with hooks
   - Cycle detection algorithms
   - Attention extraction tools

### Lens Training (Probing)

3. **`parrots/aa_fortu/aa_fortu_train_lens.py`** - Train linear probes
   - Trains linear lenses to project hidden states
   - Used for layer-wise analysis

4. **`parrots/aa_fortu/aa_fortu_train_mlp_lens.py`** - Train MLP probes
   - Trains MLP lenses for nonlinear projections
   - More expressive than linear lenses

5. **`parrots/aa_fortu/aa_fortu_train_multihead_lens.py`** - Multi-head probes
   - Trains probes for individual attention heads
   - Enables head-specific analysis

### Cycle Detection

6. **`parrots/cycle_detection.py`** - Cycle detection algorithms
   - Identifies repetitive patterns in generated text
   - Core utility for all repetition experiments

### Evaluation Utilities

7. **`parrots/nli.py`** - Natural Language Inference checking
   - Checks factual equivalence using NLI models
   - Used in slot-filling evaluation

8. **`parrots/distance.py`** - Distance metrics
   - Various distance measures for analysis
   - Embedding space analysis

### Configuration

9. **`pyproject.toml`** - Project dependencies
   - torch, transformers, transformer-lens
   - accelerate, datasets
   - matplotlib, seaborn, plotly for visualization
   - dadapy, scipy for analysis

### Shell Scripts

10. **`scripts/run_full_multihead_analysis.sh`** - SLURM batch script
    - Orchestrates multi-head analysis across layers
    - Submits parallel jobs for different checkpoints
    - Configurable for any model/checkpoint

11. **`scripts/train_mlp_lenses.sh`** - Train MLP lenses
    - Batch script for lens training across layers

---

## 3. Other Experiments (Exploratory Research)

Additional experiments not part of the main paper.

### Cycle-Attention Analysis (Breakthrough Research)

**Location:** `cycle-attention-analysis/`

This is a comprehensive investigation into causal mechanisms of repetition. See `cycle-attention-analysis/README.md` for full details.

#### Key Findings (from BREAKTHROUGH_RESEARCH_SUMMARY.md):
- **Discovered:** Semantic repetition patterns causally trigger repetitive generation
- **Best technique:** `"more and more and"` → 288 repetitions (993.3x baseline)
- **Disproved:** Direct attention interventions, newline causality hypotheses
- **Status:** Deployment-ready repetition induction techniques

#### Experiment Phases:

1. **Phase 1:** Direct Interventions (`experiments/early_interventions/`)
   - Results: 0% success (ruled out approach)
   - Scripts: `causal_attention_intervention.py`, `activation_patching_intervention.py`

2. **Phase 3:** Newline Causality (`experiments/phase3_newline_causality/`)
   - Results: 8.3% evidence (correlation ≠ causation)
   - Scripts: `investigate_newline_causality.py`, `analyze_attention_fallback.py`

3. **Phase 4:** Alternative Mechanisms (BREAKTHROUGH)
   - Results: 722.5x effectiveness
   - Discovered semantic triggers work

4. **Phase 5-6:** Exploitation & Validation
   - Results: 993.3x effectiveness
   - Deployment-ready techniques

### Entropy Analysis

1. **`logit_entropy_analysis.py`** - Logit entropy during generation
   - Analyzes prediction uncertainty
   - Compares natural vs ICL vs no-cycle conditions
   - Outputs: `logit_entropy_results.csv`

2. **`entropy_analysis.py`** - General entropy analysis
   - Broader entropy investigations

3. **`replot_entropy_no_icl.py`** - Visualization
   - Creates entropy evolution plots
   - Output: `logit_entropy_evolution_no_icl.png`

### Ablation Studies

**Location:** `ablation_results_natural_to_no_cycle_icl/`

- Tests specific interventions on HellaSwag task
- Results for different checkpoints:
  - `ablation_results_hellaswag_step1.json`
  - `ablation_results_hellaswag_step1000.json`
  - ... through step100000 and steplatest
- Analyzes ICL-first vs natural-first ordering effects

### Perturbation Analysis

1. **`parrots/perturbation_analysis.py`** - Perturbation experiments
   - Tests model robustness to input perturbations
   - Analyzes effect on repetition behavior

2. **`parrots/perturbation_analysis_distributed.py`** - Distributed version
   - Parallelized perturbation experiments

3. **`parrots/perturbation_graph.py`** - Visualization
   - Creates graphs of perturbation effects

### Distance and Representation Analysis

1. **`parrots/distance_analysis.py`** - Representation space analysis
   - Analyzes embedding distances
   - Clustering of representations

2. **`parrots/2d_mapping_*.py`** - 2D visualization
   - Projects representations to 2D
   - Multiple variants for different analyses

3. **`parrots/density_of_sequence.py`** - Sequence density analysis
   - Analyzes representation space density

### Ranking and Token Analysis

1. **`parrots/rank_distance_sequence.py`** - Rank-based analysis
   - Analyzes token ranking changes

2. **`parrots/dii_ranking_tokens.py`** - DII ranking
   - Token importance ranking

3. **`parrots/top_5_ranking_tokens.py`** - Top-k analysis
   - Analyzes top predicted tokens

### Autoprompt Experiments

1. **`parrots/autoprompt_frequency.py`** - Autoprompt analysis
   - Analyzes automatically generated prompts
   - Data: `data/autoprompts_opt1_3b_lama_parrot_list_v1.csv`

### Additional Utilities

1. **`parrots/sf_from_causal.py`** - Slot-filling from causal analysis
2. **`parrots/probability_of_sequence.py`** - Sequence probability analysis
3. **`parrots/long_term_memory.py`** - Long-term memory investigations
4. **`parrots/training_data_confidence.py`** - Training data memorization

### Shell Scripts for Other Experiments

- **`scripts/perturbated.sh`** - Run perturbation experiments
  - Supports multiple models: Pythia, OLMo, Qwen, Llama, etc.
  - Currently configured for Qwen/Qwen2.5-1.5B-Instruct

- **`scripts/slurm_sf.sh`** - Slot-filling on SLURM
  - Batch processing for multiple models/datasets

- **`scripts/pert_analy_distrib.sh`** - Distributed perturbation analysis
  - Parallel processing across compute nodes

---

## Model Support

### Currently Used (Paper):
- **EleutherAI/pythia-1.4b** - Primary model for paper experiments
- Checkpoints: step1, step1000, step5000, step10000, step100000, steplatest

### Supported in Code:
The infrastructure supports any HuggingFace causal language model:
- **Pythia family:** 70m, 1.4b, 6.9b, 12b
- **OLMo family:** allenai/OLMo-1B-hf, OLMo-1.7-7B-hf, OLMo-2-0425-1B-Instruct
- **Llama family:** meta-llama/Llama-3.2-1B, Llama-3.2-3B
- **Mistral:** mistralai/Mistral-7B-v0.3
- **Qwen:** Qwen/Qwen2.5-1.5B-Instruct, Qwen2.5-7B
- **Gemma:** google/gemma-2-2b-it

### Evidence of OLMo Support:
```python
# From scripts/slurm_sf.sh (commented out but shows capability):
# models=(
#     "EleutherAI/pythia-1.4b" 
#     "EleutherAI/pythia-6.9b"
#     "allenai/OLMo-1B-hf"
#     "allenai/OLMo-1.7-7B-hf"
# )

# From scripts/perturbated.sh (commented):
# model_name="allenai/OLMo-2-0425-1B-Instruct"
```

---

## Repeating Experiments with OLMo

### ✅ What CAN be repeated with OLMo:

1. **Slot-Filling Evaluation** (`parrots/slot_filling.py`)
   - Just change `model_name` parameter
   - Use OLMo checkpoints if available on HuggingFace

2. **Attention Analysis** (`parrots/aa_fortu/aa_fortu.py`)
   - Works with any transformer model
   - May need to adjust `max_layer_idx` for different architectures

3. **Cycle Detection** (`parrots/cycle_perturbation.py`)
   - Model-agnostic
   - Just load OLMo model

4. **Multi-head Analysis** - Infrastructure ready
   - Update shell scripts with OLMo model names
   - Adjust layer/head counts for OLMo architecture

### ⚠️ Challenges for Full Paper Replication:

1. **Training Checkpoints:**
   - Paper uses Pythia's intermediate checkpoints (step1, step1000, etc.)
   - OLMo checkpoints need to be available on HuggingFace
   - AllenAI provides some OLMo checkpoints, check availability

2. **Architecture Differences:**
   - Pythia-1.4b: 24 layers, 16 heads per layer
   - OLMo variants have different architectures
   - Need to adjust `max_layer_idx`, `n_head` parameters

3. **Tokenizer Differences:**
   - Different vocabularies may affect slot-filling results
   - Cycle detection should still work but patterns may differ

### 🔄 How to Adapt for OLMo:

#### Step 1: Test Basic Functionality
```bash
# Test slot-filling with OLMo
python -m parrots.slot_filling \
    --model-name allenai/OLMo-1B-hf \
    --data-path data/human_lama_parrots_list_v1.csv \
    --output-path outputs/OLMo-1B/
```

#### Step 2: Adjust Shell Scripts
```bash
# In scripts/run_full_multihead_analysis.sh
MODEL_NAME="allenai/OLMo-1B-hf"
# Adjust max_layer based on OLMo architecture (check model config)
for layer_idx in {0..11}; do  # OLMo-1B has 12 layers
    ...
done
```

#### Step 3: Find OLMo Checkpoints
Check HuggingFace for available OLMo checkpoints:
- allenai/OLMo-1B-hf (final checkpoint)
- allenai/OLMo-1B-0424 (intermediate checkpoints?)
- Check AllenAI documentation for checkpoint availability

#### Step 4: Run Modified Experiments
```bash
# Run cycle evolution with OLMo
# Modify run_cycle_evolution.py to use OLMo paths
# Update checkpoints list to match available OLMo checkpoints
```

### 📝 Recommended OLMo Experiment Workflow:

1. **Single Checkpoint Analysis First:**
   - Start with final OLMo model
   - Run slot-filling, cycle detection, attention analysis
   - Verify infrastructure works

2. **Compare OLMo vs Pythia:**
   - Run same experiments on both
   - Compare repetition behaviors
   - Document differences

3. **If Intermediate Checkpoints Available:**
   - Replicate full training evolution analysis
   - Create alluvial plots for OLMo
   - Compare training dynamics

4. **Novel OLMo Research:**
   - OLMo's open training data enables unique analyses
   - Could trace repetition back to specific training examples
   - Compare OLMo training dynamics to Pythia

---

## Quick Start for New Experiments

### Running Paper Experiments:
```bash
# 1. Generate perturbation data (if needed)
python -m parrots.cycle_perturbation ...

# 2. Run attention analysis
bash scripts/run_full_multihead_analysis.sh

# 3. Generate visualizations
python run_alluvial_dual.py
python run_cycle_evolution.py
python run_mlp_evolution.py
```

### Adapting for New Models:
```bash
# 1. Test basic model loading
python -c "from parrots.archs import get_model; model, tok = get_model('your-model-name')"

# 2. Run slot-filling evaluation
python -m parrots.slot_filling \
    --model-name your-model-name \
    --data-path data/human_lama_parrots_list_v1.csv \
    --output-path outputs/your-model/

# 3. Analyze results
python -m parrots.aa_fortu.aa_fortu \
    --model-name your-model-name \
    --base-path outputs/your-model/perturbations
```

---

## Citation

If using this code, please cite the repetition alluvial paper (reference in `repetition_alluvial_paper.pdf`).

## Contact

For questions about reproducing experiments or adapting for new models, contact the original authors.
