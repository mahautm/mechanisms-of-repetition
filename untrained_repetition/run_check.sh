#!/bin/bash

# Run empirical check of untrained models repetition

srun --partition=alien --qos=alien --exclude=node044 bash -lc 'source ~/.bashrc && conda activate parr && python check_untrained.py'
