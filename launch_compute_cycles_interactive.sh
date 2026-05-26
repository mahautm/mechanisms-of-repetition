#!/bin/bash
# Interactive srun launch for N tasks. Set N accordingly.

N=${1:-4}
echo "Launching $N parallel tasks..."

cd /home/mmahaut/projects/parrots
source ~/.bashrc
conda activate parr

srun \
    --partition=alien \
    --qos=alien \
    --exclude=node044 \
    -n $N \
    python compute_cycle_descriptive_stats.py \
    --rank $SLURM_PROCID \
    --n_ranks $N \
    --max_prompts 1000 \
    --max_new_tokens 512 \
    --batch_size 1
