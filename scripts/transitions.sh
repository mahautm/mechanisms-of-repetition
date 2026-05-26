#!/bin/bash

# Parameters
#SBATCH --mem=100G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044,node043
#SBATCH --error=/home/mmahaut/projects/exps/parr/%j_0_log.err
#SBATCH --job-name=extract-hlayer3
#SBATCH --output=/home/mmahaut/projects/exps/parr/%j_0_log.out
#source /etc/profile.d/zz_hpcnow-arch.sh
source ~/.bashrc

echo $SLURMD_NODENAME
conda activate parr
export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python
export PATH=$PATH:~/projects/simple-wikidata-db/
cd ~/projects/parrots/
models=("facebook/opt-1.3b")

execute_slurm() {
    sed -i "12s/^/#SBATCH --job-name=hl-$jobname\n/" slurm.sh
    sbatch slurm.sh
}

for model in "${models[@]}"
do
#     # Set jobname variable
    if [[ $model == *"nstruct"* ]]; then
        jobname="${model#*/}"
        jobname="${jobname:0:3}7i"
    else
        jobname="${model#*/}"
        jobname="${jobname:0:3}7"
    fi

    echo "Launching extraction for model $model with jobname $jobname"
    current_path=$(realpath "$0")
    echo "Current file path: $current_path"
    head -n 21 $current_path > slurm.sh
    # data_path="/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results_with_cycles.csv"
    # output_path="/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/"
    data_path="/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results_with_cycles.csv"
    output_path="/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/"

    echo "poetry run python /home/mmahaut/projects/parrots/parrots/sf_transition_test.py $data_path $model $output_path" >> slurm.sh
    execute_slurm
done
