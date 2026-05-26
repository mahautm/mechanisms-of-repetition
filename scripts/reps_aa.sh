#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044"

n_cycles=(2 3)
cycle_sizes=(2 3 5 10)
dataset_size=1000
batch_size=64
model_name="facebook/opt-1.3b"

for n_cycle in "${n_cycles[@]}"; do
    for cycle_size in "${cycle_sizes[@]}"; do
        JOB_NAME="rraa-$n_cycle-$cycle_size"
        save_path="/home/mmahaut/projects/exps/parr/$model_name/rraa"
        mkdir -p $save_path
        ERROR_LOG=$save_path/$JOB_NAME.err
        OUTPUT_LOG=$save_path/$JOB_NAME.out
        script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
which python
cd ~/projects/parrots/
poetry run python /home/mmahaut/projects/parrots/parrots/random_rep_attention_analysis.py \
    $cycle_size \
    $n_cycle \
    $dataset_size \
    $batch_size \
    $model_name \
    --save-path=$save_path
EOF
        )

        sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --gres=$GRES --qos=$QOS --exclude=$EXCLUDE
    done
done