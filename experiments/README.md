# OLMo Experiments

This directory contains ready-to-run experiments for testing OLMo models and comparing them with Pythia.

## 🚀 Quick Start

### Run Your First Test (2-5 minutes)
```bash
# Quick test with 10 samples to verify everything works
./experiments/run_olmo_experiments.sh test
```

### Run Sample Experiment (10-20 minutes)
```bash
# Standard sample with 50 examples
./experiments/run_olmo_experiments.sh sample
```

### Compare OLMo vs Pythia (20-40 minutes)
```bash
# Side-by-side comparison
./experiments/run_olmo_experiments.sh compare
```

## 📋 Available Scripts

### 1. `run_olmo_experiments.sh` - Main Entry Point
Convenient wrapper script for all experiments.

**Commands:**
- `test` - Quick verification (10 samples)
- `sample` - Standard experiment (50 samples)
- `compare` - OLMo vs Pythia comparison (30 samples each)
- `full` - Comprehensive experiment (200 samples)
- `help` - Show detailed help

**Usage:**
```bash
# Basic usage
./experiments/run_olmo_experiments.sh [command]

# With options
./experiments/run_olmo_experiments.sh sample --sample-size 100

# Custom models
./experiments/run_olmo_experiments.sh compare \
    --olmo-model allenai/OLMo-7B-hf \
    --pythia-model EleutherAI/pythia-6.9b
```

### 2. `olmo_sample_experiment.py` - Single Model Analysis
Complete experimental pipeline for one model.

**What it does:**
1. Slot-filling evaluation (factual knowledge)
2. Cycle detection (repetition behavior)
3. Attention pattern analysis
4. Visualization generation
5. Automated reporting

**Direct usage:**
```bash
python experiments/olmo_sample_experiment.py \
    --model allenai/OLMo-1B-hf \
    --sample-size 50 \
    --output-dir outputs/my_experiment
```

**Options:**
- `--model` - Model to analyze (default: allenai/OLMo-1B-hf)
- `--sample-size` - Number of examples (default: 50)
- `--output-dir` - Where to save results
- `--device` - cuda or cpu (default: cuda)
- `--data-path` - Custom dataset path

### 3. `compare_olmo_pythia.py` - Model Comparison
Side-by-side comparison of two or more models.

**What it does:**
- Runs full experiments on multiple models
- Creates comparative visualizations
- Generates comparison report with winner analysis
- Identifies behavioral differences

**Direct usage:**
```bash
python experiments/compare_olmo_pythia.py \
    --sample-size 30 \
    --olmo-model allenai/OLMo-1B-hf \
    --pythia-model EleutherAI/pythia-1.4b
```

## 📊 Experiments Included

### Experiment 1: Slot-Filling Evaluation
Tests factual knowledge using LAMA dataset prompts like:
- "The capital of France is ___"
- "Barack Obama was born in ___"

**Metrics:**
- **Direct Follow**: Answer is the immediate next token
- **Exact Match**: Answer appears anywhere in generation
- **NLI Factual**: Answer is factually equivalent (uses NLI model)

### Experiment 2: Cycle Detection
Generates longer sequences and detects repetitive patterns.

**Metrics:**
- Repetition rate (% of samples with cycles)
- Average cycle length (in tokens)
- Maximum cycle count

**Example cycle:**
```
"The cat sat on the cat sat on the cat sat on the..."
```

### Experiment 3: Attention Analysis
Analyzes attention patterns across layers and heads.

**Metrics:**
- Attention entropy (how focused/diffuse)
- Maximum attention values
- Layer-wise patterns
- Head-wise specialization

**Compares:**
- Samples with cycles vs without
- Different layers (early vs late)
- Critical attention heads

### Experiment 4: Visualizations
Auto-generates plots:
- Slot-filling performance bars
- Cycle statistics histograms
- Attention heatmaps
- Layer-wise comparisons
- Model architecture comparisons (for multi-model)

## 📁 Output Structure

After running experiments, you'll find:

```
outputs/
├── olmo_sample_experiment/          # Single model results
│   ├── data/
│   │   ├── sample_data.csv          # Input samples used
│   │   ├── slot_filling_results.csv # Detailed results
│   │   ├── cycle_detection_results.csv
│   │   ├── attention_patterns.csv
│   │   └── layer_attention_stats.csv
│   ├── plots/
│   │   ├── slot_filling_performance.png
│   │   ├── cycle_statistics.png
│   │   ├── attention_entropy_heatmap.png
│   │   └── attention_comparison.png
│   ├── logs/
│   ├── experiment_report.md         # Human-readable summary
│   └── experiment_results.json      # Machine-readable results
│
└── olmo_pythia_comparison/          # Comparison results
    ├── OLMo-1B/                     # Individual OLMo results
    ├── Pythia-1.4B/                 # Individual Pythia results
    ├── comparison_plots/
    │   ├── slot_filling_comparison.png
    │   ├── repetition_comparison.png
    │   └── architecture_comparison.png
    ├── comparison_report.md         # Side-by-side analysis
    └── comparison_results.json
```

## 🔧 Requirements

### Hardware
- **Minimum**: 8GB GPU RAM (for 1B models)
- **Recommended**: 16GB+ GPU RAM
- **CPU fallback**: Works but much slower

### Software
- Python 3.9+
- CUDA (for GPU acceleration)
- Poetry (for dependency management)

### Dependencies
All handled by Poetry:
```bash
poetry install
```

Key packages:
- `torch` - PyTorch
- `transformers` - HuggingFace models
- `pandas` - Data handling
- `matplotlib`, `seaborn` - Visualizations
- `tqdm` - Progress bars

## 🎯 Use Cases

### 1. Quick Verification
**Goal**: Check if OLMo works with your setup
```bash
./experiments/run_olmo_experiments.sh test
```
**Time**: 2-5 minutes

### 2. Model Evaluation
**Goal**: Understand OLMo's behavior on your task
```bash
./experiments/run_olmo_experiments.sh sample --sample-size 100
```
**Time**: 15-30 minutes

### 3. Model Selection
**Goal**: Choose between OLMo and Pythia
```bash
./experiments/run_olmo_experiments.sh compare
```
**Time**: 20-40 minutes

### 4. Research Analysis
**Goal**: Detailed analysis for paper/report
```bash
./experiments/run_olmo_experiments.sh full
```
**Time**: 1-2 hours

### 5. Custom Comparison
**Goal**: Test specific model variants
```bash
python experiments/compare_olmo_pythia.py \
    --olmo-model allenai/OLMo-7B-hf \
    --pythia-model EleutherAI/pythia-6.9b \
    --sample-size 50
```

## 💡 Tips & Tricks

### Running on CPU
If no GPU available:
```bash
./experiments/run_olmo_experiments.sh test --device cpu
```

### Smaller Memory Footprint
```bash
# Use smaller sample
./experiments/run_olmo_experiments.sh sample --sample-size 20

# Or use quantization (edit Python script to add):
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

### Different Models
Try other OLMo variants:
```bash
# Larger model
./experiments/run_olmo_experiments.sh sample \
    --model allenai/OLMo-7B-hf

# Instruction-tuned
./experiments/run_olmo_experiments.sh sample \
    --model allenai/OLMo-2-0425-1B-Instruct
```

### Custom Data
Use your own dataset:
```bash
python experiments/olmo_sample_experiment.py \
    --data-path path/to/your/data.csv \
    --sample-size 50
```

Your CSV should have columns:
- `sub_label` - Input prompts
- `obj_label` - Expected outputs (optional)

### Batch Processing
Run multiple experiments in sequence:
```bash
for model in allenai/OLMo-1B-hf allenai/OLMo-7B-hf; do
    ./experiments/run_olmo_experiments.sh sample \
        --model $model \
        --output-dir outputs/$(basename $model)
done
```

## 🐛 Troubleshooting

### Out of Memory
**Error**: `CUDA out of memory`

**Solutions:**
1. Reduce sample size: `--sample-size 20`
2. Use CPU: `--device cpu`
3. Use smaller model: `allenai/OLMo-1B-hf` instead of 7B
4. Enable quantization (modify script)

### Model Not Found
**Error**: `Model allenai/OLMo-1B-hf not found`

**Solutions:**
1. Check internet connection
2. Login to HuggingFace: `huggingface-cli login`
3. Verify model exists: https://huggingface.co/allenai/OLMo-1B-hf

### Slow Generation
**Issue**: Taking very long time

**Solutions:**
1. Use GPU if available
2. Reduce `max_new_tokens` in script
3. Use smaller batch size
4. Check if model is in CPU mode unintentionally

### Import Errors
**Error**: `ModuleNotFoundError: No module named 'parrots'`

**Solutions:**
1. Run from project root: `cd /home/mmahaut/projects/parrots`
2. Install dependencies: `poetry install`
3. Activate environment: `poetry shell`

## 📚 Next Steps

After running experiments:

1. **Review Results**
   ```bash
   # View the report
   cat outputs/olmo_sample_experiment/experiment_report.md
   
   # Check plots
   ls outputs/olmo_sample_experiment/plots/
   ```

2. **Analyze Data**
   ```python
   import pandas as pd
   
   # Load results
   df = pd.read_csv('outputs/olmo_sample_experiment/data/slot_filling_results.csv')
   
   # Analyze
   print(df[['input', 'expected', 'generated']].head())
   print(f"Accuracy: {df['exact_match'].mean():.1%}")
   ```

3. **Scale Up**
   - Increase sample size
   - Try different models
   - Run on full dataset

4. **Customize**
   - Modify experiment scripts
   - Add new metrics
   - Create custom visualizations

5. **Compare More Models**
   ```bash
   # Add more models to compare
   # Edit compare_olmo_pythia.py and add to models dict
   ```

## 🔗 Related Documentation

- **[OLMO_ADAPTATION_GUIDE.md](../OLMO_ADAPTATION_GUIDE.md)** - Complete guide for adapting paper experiments
- **[EXPERIMENT_ORGANIZATION.md](../EXPERIMENT_ORGANIZATION.md)** - Overview of all experiments
- **[QUICK_REFERENCE.md](../QUICK_REFERENCE.md)** - Quick command reference
- **[README.md](../README.md)** - Main project README

## 🤝 Contributing

To add new experiments:

1. Create new Python script in `experiments/`
2. Follow the pattern from `olmo_sample_experiment.py`
3. Add command to `run_olmo_experiments.sh`
4. Update this README

## 📝 Example Workflow

Complete workflow from scratch:

```bash
# 1. Setup
cd /home/mmahaut/projects/parrots
poetry install
poetry shell

# 2. Quick test (verify everything works)
./experiments/run_olmo_experiments.sh test

# 3. Run sample experiment
./experiments/run_olmo_experiments.sh sample

# 4. Review results
cat outputs/olmo_sample_experiment/experiment_report.md
ls outputs/olmo_sample_experiment/plots/

# 5. Compare models
./experiments/run_olmo_experiments.sh compare

# 6. Check comparison
cat outputs/olmo_pythia_comparison/comparison_report.md

# 7. Analyze data (optional)
python
>>> import pandas as pd
>>> df = pd.read_csv('outputs/olmo_sample_experiment/data/slot_filling_results.csv')
>>> df.describe()

# 8. Scale up (if results look good)
./experiments/run_olmo_experiments.sh full
```

## 📧 Support

For issues:
1. Check troubleshooting section above
2. Review error messages in logs
3. Check main documentation files
4. Open an issue with error details

---

**Ready to start?** Run your first test:
```bash
./experiments/run_olmo_experiments.sh test
```
