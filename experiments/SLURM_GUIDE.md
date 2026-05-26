# OLMo Experiments on SLURM - Quick Guide

## 🚀 Quick Start

### 1. Run Quick Test
```bash
cd /home/mmahaut/projects/parrots
./experiments/run_olmo_experiments.sh test
```

This submits a test job to verify everything works.

### 2. Monitor Your Job
```bash
# Check status
squeue -u $USER

# Watch output
tail -f logs/olmo_test_JOBID.out

# Check errors
tail -f logs/olmo_test_JOBID.err
```

### 3. Run Full Experiment
```bash
# Sample experiment (50 examples)
./experiments/run_olmo_experiments.sh sample

# Comparison (OLMo vs Pythia)
./experiments/run_olmo_experiments.sh compare
```

## 📋 Available SLURM Scripts

### Direct SLURM Submission

All scripts are in `experiments/` directory:

1. **`slurm_olmo_test.sh`** - Quick test (10 samples, ~5-10 min)
```bash
sbatch experiments/slurm_olmo_test.sh
```

2. **`slurm_olmo_sample.sh`** - Sample experiment (50 samples, ~20-40 min)
```bash
# Default
sbatch experiments/slurm_olmo_sample.sh

# Custom parameters
sbatch --export=MODEL=allenai/OLMo-7B-hf,SAMPLE_SIZE=100 \
    experiments/slurm_olmo_sample.sh
```

3. **`slurm_olmo_compare.sh`** - Model comparison (~40-80 min)
```bash
# Default (OLMo-1B vs Pythia-1.4B)
sbatch experiments/slurm_olmo_compare.sh

# Custom models
sbatch --export=OLMO_MODEL=allenai/OLMo-7B-hf,PYTHIA_MODEL=EleutherAI/pythia-6.9b \
    experiments/slurm_olmo_compare.sh
```

### Using the Launcher Script

The launcher provides easier interface:

```bash
# Quick test
./experiments/run_olmo_experiments.sh test

# Sample with options
./experiments/run_olmo_experiments.sh sample --sample-size 100

# Compare models
./experiments/run_olmo_experiments.sh compare \
    --olmo-model allenai/OLMo-7B-hf \
    --pythia-model EleutherAI/pythia-6.9b

# Check status
./experiments/run_olmo_experiments.sh status

# Cancel jobs
./experiments/run_olmo_experiments.sh cancel
```

## ⚙️ Configuration

### SLURM Resource Requirements

Default resources (adjust in SLURM scripts):

**Test job:**
- Time: 30 minutes
- Memory: 32GB
- GPU: 1
- CPUs: 4

**Sample/Compare jobs:**
- Time: 2-4 hours
- Memory: 64GB
- GPU: 1
- CPUs: 8

### Customizing Resources

Edit the `#SBATCH` directives in SLURM scripts:

```bash
# Increase time limit
#SBATCH --time=08:00:00

# More memory
#SBATCH --mem=128G

# Specific GPU type
#SBATCH --gres=gpu:a100:1

# Specific partition
#SBATCH --partition=gpu-long
```

### Environment Variables

Pass custom parameters:

```bash
# Model selection
sbatch --export=MODEL=allenai/OLMo-7B-hf experiments/slurm_olmo_sample.sh

# Sample size
sbatch --export=SAMPLE_SIZE=200 experiments/slurm_olmo_sample.sh

# Output directory
sbatch --export=OUTPUT_DIR=outputs/my_experiment experiments/slurm_olmo_sample.sh

# Multiple parameters
sbatch --export=MODEL=allenai/OLMo-1B-hf,SAMPLE_SIZE=100,OUTPUT_DIR=outputs/test \
    experiments/slurm_olmo_sample.sh
```

## 📊 Monitoring Jobs

### Check Job Status
```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j JOBID

# Detailed format
squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
```

### View Logs in Real-Time
```bash
# Output
tail -f logs/olmo_test_JOBID.out

# Errors
tail -f logs/olmo_test_JOBID.err

# Both
tail -f logs/olmo_test_JOBID.{out,err}
```

### Job History
```bash
# Recent jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed

# Specific job details
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
```

### Cancel Jobs
```bash
# Specific job
scancel JOBID

# All your jobs
scancel -u $USER

# By job name
scancel -n olmo_test

# All OLMo jobs (using launcher)
./experiments/run_olmo_experiments.sh cancel
```

## 🔍 Debugging

### Job Failed - Check Logs
```bash
# Last 50 lines of error log
tail -50 logs/olmo_test_JOBID.err

# Search for errors
grep -i error logs/olmo_test_JOBID.err

# Check Python exceptions
grep -A 10 "Traceback" logs/olmo_test_JOBID.err
```

### Common Issues

#### 1. Job Pending
```bash
# Check why pending
squeue -j JOBID -o "%.18i %.9P %.20j %.8T %.10M %.10l %R"

# Check partition availability
sinfo -p gpu
```

**Solutions:**
- Wait if "Resources" or "(Priority)"
- Check if requesting unavailable resources
- Try different partition: `--partition=gpu-short`

#### 2. Out of Memory
```bash
# Check memory usage
sacct -j JOBID --format=JobID,MaxRSS,MaxVMSize
```

**Solutions:**
- Reduce sample size in script
- Request more memory: `#SBATCH --mem=128G`
- Use smaller model

#### 3. GPU Not Available
```bash
# Check GPU allocation
scontrol show job JOBID | grep GPU
```

**Solutions:**
- Check partition has GPUs: `sinfo -p gpu`
- Verify GPU request: `#SBATCH --gres=gpu:1`
- Try different GPU partition

#### 4. Module/Environment Errors
```bash
# Check if environment activates
srun --pty bash
source $(poetry env info --path)/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
exit
```

**Solutions:**
- Ensure Poetry installed
- Run `poetry install` in login node first
- Check CUDA modules loaded

#### 5. Model Download Issues
```bash
# Test model access
srun --pty bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('allenai/OLMo-1B-hf')"
```

**Solutions:**
- Login to HuggingFace: `huggingface-cli login`
- Check internet on compute nodes
- Pre-download model on login node

## 📁 Output Files

After job completes, results in:

```
outputs/
├── olmo_sample_experiment/
│   ├── data/
│   │   ├── slot_filling_results.csv
│   │   ├── cycle_detection_results.csv
│   │   └── attention_patterns.csv
│   ├── plots/
│   │   ├── slot_filling_performance.png
│   │   ├── cycle_statistics.png
│   │   └── attention_entropy_heatmap.png
│   ├── experiment_report.md
│   └── experiment_results.json
└── logs/
    ├── olmo_test_JOBID.out
    └── olmo_test_JOBID.err
```

## 🎯 Typical Workflow

### Complete Experiment Flow
```bash
# 1. Navigate to project
cd /home/mmahaut/projects/parrots

# 2. Run quick test
./experiments/run_olmo_experiments.sh test
# Note the job ID

# 3. Monitor test
tail -f logs/olmo_test_JOBID.out

# 4. Check test results
cat outputs/olmo_quick_test/experiment_report.md

# 5. If successful, run sample
./experiments/run_olmo_experiments.sh sample

# 6. Monitor sample
./experiments/run_olmo_experiments.sh status
tail -f logs/olmo_sample_JOBID.out

# 7. Check results
cat outputs/olmo_sample_experiment/experiment_report.md
ls outputs/olmo_sample_experiment/plots/

# 8. Run comparison
./experiments/run_olmo_experiments.sh compare

# 9. View comparison results
cat outputs/olmo_pythia_comparison/comparison_report.md
```

### Batch Experiments
```bash
# Submit multiple experiments
for size in 30 50 100; do
    sbatch --export=SAMPLE_SIZE=$size,OUTPUT_DIR=outputs/olmo_sample_${size} \
        experiments/slurm_olmo_sample.sh
    sleep 2
done

# Monitor all
watch -n 5 'squeue -u $USER'
```

## 💡 Tips

### 1. Test Before Large Runs
Always run test first to catch issues early:
```bash
./experiments/run_olmo_experiments.sh test
```

### 2. Use Job Arrays for Multiple Experiments
Create a job array script:
```bash
#SBATCH --array=1-5
MODEL=allenai/OLMo-1B-hf
SAMPLE_SIZE=$((SLURM_ARRAY_TASK_ID * 20))
```

### 3. Email Notifications
Add to SLURM scripts:
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com
```

### 4. Save Output in Real-Time
For long jobs, monitor progress:
```bash
watch -n 30 tail -20 logs/olmo_sample_JOBID.out
```

### 5. Resource Estimation
Check previous job resources:
```bash
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
```

Use this to request appropriate resources for future jobs.

## 🔧 Customization

### Add Your Own Experiment

1. Create SLURM script `experiments/slurm_my_experiment.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=my_exp
#SBATCH --output=logs/my_exp_%j.out
#SBATCH --error=logs/my_exp_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

cd /home/mmahaut/projects/parrots
source $(poetry env info --path)/bin/activate

srun python experiments/my_experiment.py "$@"
```

2. Submit:
```bash
sbatch experiments/slurm_my_experiment.sh
```

### Modify Existing Scripts

Edit SLURM scripts to adjust:
- Time limits
- Memory requirements
- GPU types
- Partitions
- Default parameters

## 📚 Additional Resources

- **SLURM Documentation**: https://slurm.schedmd.com/
- **Project Documentation**: 
  - `experiments/README.md`
  - `docs/02-USING-OLMO.md`
- **Python Scripts**: `experiments/*.py`

## 🆘 Getting Help

```bash
# Launcher help
./experiments/run_olmo_experiments.sh help

# Python script help
python experiments/olmo_sample_experiment.py --help

# SLURM help
man sbatch
man squeue
man scancel
```

---

**Quick Command Reference:**

```bash
# Submit
sbatch experiments/slurm_olmo_test.sh

# Status
squeue -u $USER

# Monitor
tail -f logs/olmo_test_JOBID.out

# Cancel
scancel JOBID

# Results
cat outputs/olmo_quick_test/experiment_report.md
```
