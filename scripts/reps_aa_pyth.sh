#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:0"
QOS="alien"
EXCLUDE="node044"

# n_cycles=(3)
cycle_sizes=(3 5 10)
dataset_size=1000
batch_size=64
redownload=false
# model_name="facebook/opt-1.3b"
# pythia_checkpoint_roots=("EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-1.4B" "EleutherAI/pythia-6.9B" "EleutherAI/pythia-12B")
pythia_checkpoint_roots=("EleutherAI/pythia-1.4B")

# checkpoint_list=("step1" "step1000" "step5000" "step7000" "step10000" "step15000" "step20000" "step50000" "step100000" )
checkpoint_list=("step100000")
for model_name in "${pythia_checkpoint_roots[@]}"; do
    for checkpoint in "${checkpoint_list[@]}"; do
        if [ "$redownload" = true ]; then
            echo "Downloading model $model_name with checkpoint $checkpoint"
            python -c "from transformers import AutoModel; AutoModel.from_pretrained('$model_name', revision='$checkpoint', force_download=True)"
        fi
        # for n_cycle in "${n_cycles[@]}"; do
        for cycle_size in "${cycle_sizes[@]}"; do
            max_tok_idx=$(( 2 * $cycle_size))
            for tok_idx in $(seq 0 $max_tok_idx); do
                JOB_NAME="rraa-$tok_idx-$cycle_size-$checkpoint"
                save_path="/home/mmahaut/projects/exps/parr/$model_name/hprraa_th0.01"
                mkdir -p $save_path
                ERROR_LOG="$save_path/$JOB_NAME.err"
                OUTPUT_LOG="$save_path/$JOB_NAME.out"
                script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
which python
cd ~/projects/parrots/
poetry run python /home/mmahaut/projects/parrots/parrots/rraa_first_loop_human.py \
    $cycle_size \
    $tok_idx \
    $dataset_size \
    $batch_size \
    $model_name \
    --revision $checkpoint \
    --save-path=$save_path \
    --n-devices 1
EOF
        )
                sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES
            done
        done
    done
done