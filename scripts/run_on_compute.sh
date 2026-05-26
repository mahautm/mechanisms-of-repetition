#!/usr/bin/env bash
# Wrapper to run commands on a compute node (srun) with the project's conda env activated.
# Usage: ./scripts/run_on_compute.sh python compute_cycle_descriptive_stats.py --rank 0 ...

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <command...>"
  exit 1
fi

srun --partition=alien --qos=alien --exclude=node044 bash -lc "source ~/.bashrc && conda activate parr && $*"
