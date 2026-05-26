#!/bin/bash
# OLMo Experiments Launcher for SLURM
# ====================================
# 
# This script submits OLMo experiments to SLURM queue.
#
# Usage:
#   ./experiments/run_olmo_experiments.sh [command] [options]
#
# Commands:
#   test        - Quick test with tiny sample (10 examples)
#   sample      - Run standard sample experiment (50 examples)
#   compare     - Compare OLMo vs Pythia (30 examples each)
#   full        - Run full experiment (200+ examples)
#   status      - Check job status
#   cancel      - Cancel running jobs
#   help        - Show this help message

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/mmahaut/projects/parrots"
cd "$PROJECT_ROOT"

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check SLURM
    if ! command -v sbatch &> /dev/null; then
        print_error "SLURM not found! This script requires SLURM."
        print_warning "For non-SLURM systems, run Python scripts directly:"
        echo "  python experiments/olmo_sample_experiment.py --help"
        exit 1
    fi
    print_success "SLURM available"
    
    # Check if in project directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "Not in project root directory!"
        echo "Please run from: /home/mmahaut/projects/parrots"
        exit 1
    fi
    
    # Check data file
    if [ ! -f "data/human_lama_parrots_list_v1.csv" ]; then
        print_error "Data file not found: data/human_lama_parrots_list_v1.csv"
        exit 1
    fi
    print_success "Data file found"
    
    # Create logs directory
    mkdir -p logs
    print_success "Logs directory ready"
    
    echo ""
}

# Command implementations
run_test() {
    print_header "Submitting Quick Test (10 samples)"
    echo "This is a quick test to verify everything works."
    echo "Expected time: ~5-10 minutes (queued + runtime)"
    echo ""
    
    JOB_ID=$(sbatch --parsable experiments/slurm_olmo_test.sh)
    
    if [ $? -eq 0 ]; then
        print_success "Test job submitted!"
        echo ""
        echo "Job ID: $JOB_ID"
        echo "Monitor: squeue -j $JOB_ID"
        echo "Output: tail -f logs/olmo_test_${JOB_ID}.out"
        echo "Errors: tail -f logs/olmo_test_${JOB_ID}.err"
        echo ""
        echo "Cancel job: scancel $JOB_ID"
    fi
}

run_sample() {
    print_header "Submitting Sample Experiment (50 samples)"
    echo "Standard sample experiment with OLMo-1B."
    echo "Expected time: ~20-40 minutes (queued + runtime)"
    echo ""
    
    # Parse optional arguments
    MODEL="allenai/OLMo-1B-hf"
    SAMPLE_SIZE=50
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --sample-size)
                SAMPLE_SIZE="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    JOB_ID=$(sbatch --parsable \
        --export=MODEL="$MODEL",SAMPLE_SIZE="$SAMPLE_SIZE" \
        experiments/slurm_olmo_sample.sh)
    
    if [ $? -eq 0 ]; then
        print_success "Sample job submitted!"
        echo ""
        echo "Job ID: $JOB_ID"
        echo "Model: $MODEL"
        echo "Sample size: $SAMPLE_SIZE"
        echo ""
        echo "Monitor: squeue -j $JOB_ID"
        echo "Output: tail -f logs/olmo_sample_${JOB_ID}.out"
        echo "Errors: tail -f logs/olmo_sample_${JOB_ID}.err"
        echo ""
        echo "Cancel job: scancel $JOB_ID"
    fi
}

run_compare() {
    print_header "Submitting OLMo vs Pythia Comparison (30 samples each)"
    echo "Side-by-side comparison of OLMo-1B and Pythia-1.4B."
    echo "Expected time: ~40-80 minutes (queued + runtime)"
    echo ""
    
    # Parse optional arguments
    SAMPLE_SIZE=30
    OLMO_MODEL="allenai/OLMo-1B-hf"
    PYTHIA_MODEL="EleutherAI/pythia-1.4b"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --sample-size)
                SAMPLE_SIZE="$2"
                shift 2
                ;;
            --olmo-model)
                OLMO_MODEL="$2"
                shift 2
                ;;
            --pythia-model)
                PYTHIA_MODEL="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    JOB_ID=$(sbatch --parsable \
        --export=SAMPLE_SIZE="$SAMPLE_SIZE",OLMO_MODEL="$OLMO_MODEL",PYTHIA_MODEL="$PYTHIA_MODEL" \
        experiments/slurm_olmo_compare.sh)
    
    if [ $? -eq 0 ]; then
        print_success "Comparison job submitted!"
        echo ""
        echo "Job ID: $JOB_ID"
        echo "Sample size: $SAMPLE_SIZE"
        echo "OLMo: $OLMO_MODEL"
        echo "Pythia: $PYTHIA_MODEL"
        echo ""
        echo "Monitor: squeue -j $JOB_ID"
        echo "Output: tail -f logs/olmo_compare_${JOB_ID}.out"
        echo "Errors: tail -f logs/olmo_compare_${JOB_ID}.err"
        echo ""
        echo "Cancel job: scancel $JOB_ID"
    fi
}

run_full() {
    print_header "Submitting Full Experiment (200 samples)"
    echo "⚠️  WARNING: This will take significant time and compute!"
    echo "Expected time: 1-2 hours (queued + runtime)"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    JOB_ID=$(sbatch --parsable \
        --export=MODEL="allenai/OLMo-1B-hf",SAMPLE_SIZE=200 \
        experiments/slurm_olmo_sample.sh)
    
    if [ $? -eq 0 ]; then
        print_success "Full experiment job submitted!"
        echo ""
        echo "Job ID: $JOB_ID"
        echo ""
        echo "Monitor: squeue -j $JOB_ID"
        echo "Output: tail -f logs/olmo_sample_${JOB_ID}.out"
    fi
}

show_status() {
    print_header "SLURM Job Status"
    
    echo "Your OLMo experiment jobs:"
    squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.6D %R" | grep -E "JOB|olmo" || echo "No OLMo jobs found"
    echo ""
    
    echo "Recent job outputs:"
    ls -lht logs/olmo_*.out 2>/dev/null | head -5 || echo "No output files yet"
}

cancel_jobs() {
    print_header "Cancel OLMo Jobs"
    
    # Get OLMo job IDs
    JOBS=$(squeue -u $USER -o "%.18i %.20j" | grep olmo | awk '{print $1}')
    
    if [ -z "$JOBS" ]; then
        echo "No OLMo jobs found to cancel"
        return
    fi
    
    echo "Found OLMo jobs:"
    squeue -u $USER -o "%.18i %.9P %.20j %.8T" | grep -E "JOB|olmo"
    echo ""
    
    read -p "Cancel all these jobs? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for JOB_ID in $JOBS; do
            scancel $JOB_ID
            echo "Cancelled job: $JOB_ID"
        done
        print_success "All OLMo jobs cancelled"
    else
        echo "Cancelled."
    fi
}

show_help() {
    cat << EOF
${BLUE}OLMo Sample Experiments - SLURM Launcher${NC}

${YELLOW}Usage:${NC}
    ./experiments/run_olmo_experiments.sh [command] [options]

${YELLOW}Commands:${NC}
    ${GREEN}test${NC}        Quick test with 10 samples (~5-10 min)
                Verifies everything works before larger runs
                Submits job to SLURM queue
                
    ${GREEN}sample${NC}      Standard sample with 50 samples (~20-40 min)
                Good for initial exploration and testing
                Options: --model MODEL --sample-size N
                
    ${GREEN}compare${NC}     Compare OLMo vs Pythia (~40-80 min)
                Runs samples through both models
                Options: --sample-size N --olmo-model M --pythia-model M
                
    ${GREEN}full${NC}        Full experiment with 200 samples (~1-2 hours)
                More comprehensive analysis
                ⚠️  Requires significant compute time
    
    ${GREEN}status${NC}      Check status of running jobs
                Shows your OLMo experiment jobs in queue
    
    ${GREEN}cancel${NC}      Cancel running OLMo jobs
                Cancels all your OLMo experiment jobs
                
    ${GREEN}help${NC}        Show this help message

${YELLOW}Options:${NC}
    You can pass additional options to the Python scripts:
    
    --model MODEL_NAME        Use different model
    --sample-size N           Override sample size
    --output-dir DIR          Change output directory
    --device cpu              Force CPU usage

${YELLOW}Examples:${NC}
    # Quick test
    ./experiments/run_olmo_experiments.sh test
    
    # Sample experiment with custom size
    ./experiments/run_olmo_experiments.sh sample --sample-size 100
    
    # Compare with different models
    ./experiments/run_olmo_experiments.sh compare \\
        --olmo-model allenai/OLMo-7B-hf \\
        --pythia-model EleutherAI/pythia-6.9b
    
    # Check job status
    ./experiments/run_olmo_experiments.sh status
    
    # Monitor job output (use job ID from submit)
    tail -f logs/olmo_test_12345.out

${YELLOW}Output:${NC}
    Results are saved to outputs/ directory:
    - Data files (CSV, JSON)
    - Visualizations (PNG plots)
    - Reports (Markdown summaries)

${YELLOW}Requirements:${NC}
    - SLURM cluster access
    - Python 3.9+ with Poetry
    - GPU node with CUDA
    - ~8GB GPU RAM for 1B models
    - ~20GB GPU RAM for 7B models

${YELLOW}SLURM Commands:${NC}
    Check job status:    squeue -u \$USER
    Cancel job:          scancel JOB_ID
    View output:         tail -f logs/olmo_*_JOB_ID.out
    View errors:         tail -f logs/olmo_*_JOB_ID.err
    Check account:       sacct -j JOB_ID

${YELLOW}Troubleshooting:${NC}
    Job pending:
        - Check queue: squeue -u \$USER
        - Check partitions: sinfo
    
    Out of memory:
        - Reduce sample size: --sample-size 20
        - Request more memory in SLURM scripts
    
    Model not found:
        - May need: huggingface-cli login
        - Check internet on compute nodes

For detailed documentation, see:
    - experiments/README.md
    - docs/02-USING-OLMO.md
    - docs/01-EXPERIMENTS.md

${YELLOW}Direct SLURM Usage:${NC}
    You can also submit jobs directly:
    
    sbatch experiments/slurm_olmo_test.sh
    sbatch --export=SAMPLE_SIZE=100 experiments/slurm_olmo_sample.sh
    sbatch experiments/slurm_olmo_compare.sh

EOF
}

# Main command dispatcher
COMMAND="${1:-help}"

case "$COMMAND" in
    test)
        shift
        check_dependencies
        run_test "$@"
        ;;
    sample)
        shift
        check_dependencies
        run_sample "$@"
        ;;
    compare)
        shift
        check_dependencies
        run_compare "$@"
        ;;
    full)
        shift
        check_dependencies
        run_full "$@"
        ;;
    status)
        show_status
        ;;
    cancel)
        cancel_jobs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        echo "Run './experiments/run_olmo_experiments.sh help' for usage information"
        exit 1
        ;;
esac
