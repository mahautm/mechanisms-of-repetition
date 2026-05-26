#!/bin/bash
#SBATCH --job-name=test_compute
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_%J.out

mkdir -p logs

srun conda run -n parr python -c "import torch; print('GPU available:', torch.cuda.is_available())"
