#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044"

n_cycles=(2 3 4)
cycle_sizes=(2 3 5 10)
dataset_size=1000
batch_size=64
redownload=true
# model_name="facebook/opt-1.3b"
# pythia_checkpoint_roots=("EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-1.4B" "EleutherAI/pythia-6.9B" "EleutherAI/pythia-12B")
pythia_checkpoint_roots=("EleutherAI/pythia-12B")
checkpoint_list=("step1" "step1000" "step5000" "step7000" "step10000" "step15000" "step20000" "step50000" "step100000" )

for model_name in "${pythia_checkpoint_roots[@]}"; do
    for checkpoint in "${checkpoint_list[@]}"; do
        if [ "$redownload" = true ]; then
            echo "Downloading model $model_name with checkpoint $checkpoint"
            python -c "from transformers import AutoModel; AutoModel.from_pretrained('$model_name', revision='$checkpoint')"
        fi
        for n_cycle in "${n_cycles[@]}"; do
            for cycle_size in "${cycle_sizes[@]}"; do
                JOB_NAME="rr-$n_cycle-$cycle_size-$checkpoint"
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
    --revision $checkpoint \
    --deactivate-tqdm \
    --use-bnb
EOF
                )

                sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --gres=$GRES --qos=$QOS --exclude=$EXCLUDE
            done
        done
    done
done