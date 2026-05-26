#!/bin/bash

# Parameters
#SBATCH --mem=200G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044,node043
#SBATCH --error=/home/mmahaut/projects/exps/parr/at_pile_%j_0_log.err
#SBATCH --job-name=at_pile
#SBATCH --output=/home/mmahaut/projects/exps/parr/at_pile_%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate parr
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
cd ~/projects/parrots/
models=("facebook/opt-1.3b")

poetry run python /home/mmahaut/projects/parrots/parrots/attention_through_pile.py
