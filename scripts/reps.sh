#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044"

n_cycles=(1 2 3)
cycle_sizes=(2 3 5 10)
dataset_size=1000
batch_size=64
model_name="facebook/opt-1.3b"
# pythia_checkpoint_roots=("EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-1.4B" "EleutherAI/pythia-6.9B" "EleutherAI/pythia-12B")
# checkpoint_list = ("step1" "step2" "step512" "step1000" "step10000" "step100000")
deactivate_tqdm="--deactivate-tqdm"

for n_cycle in "${n_cycles[@]}"; do
    for cycle_size in "${cycle_sizes[@]}"; do
        JOB_NAME="rr-$n_cycle-$cycle_size"
        mkdir -p "/home/mmahaut/projects/exps/parr/$model_name"
        ERROR_LOG="/home/mmahaut/projects/exps/parr/$model_name/$JOB_NAME.err"
        OUTPUT_LOG="/home/mmahaut/projects/exps/parr/$model_name/$JOB_NAME.out"
        script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
which python
cd ~/projects/parrots/
poetry run python /home/mmahaut/projects/parrots/parrots/random_rep.py \
    $cycle_size \
    $n_cycle \
    $dataset_size \
    $batch_size \
    $model_name \
    --deactivate-tqdm
EOF
        )

        sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --gres=$GRES --qos=$QOS --exclude=$EXCLUDE
    done
done