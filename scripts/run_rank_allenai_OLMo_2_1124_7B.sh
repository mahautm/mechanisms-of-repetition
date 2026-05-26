#!/bin/bash
#SBATCH --job-name=cycle_allenai_OLMo_2_1124_7B
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --array=0-9
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=06:00:00

source ~/.bashrc
conda activate parr

python compute_cycle_descriptive_stats.py   --n_ranks 10   --rank $SLURM_ARRAY_TASK_ID   --max_prompts 1000   --prompt_size 512   --max_new_tokens 1000   --batch_size 8   --models allenai/OLMo-2-1124-7B   --allow_downloads   --device cuda   --device_map auto
