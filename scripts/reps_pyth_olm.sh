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
# pythia_checkpoint_roots=("allenai/OLMo-2-1124-7B" "allenai/OLMo-2-1124-13B")
pythia_checkpoint_roots=("allenai/OLMo-2-1124-13B")
# checkpoint_list=("stage1-step150-tokens1B" "stage1-step1000-tokens5B" "stage1-step5000-tokens21B" "stage1-step7000-tokens30B" "stage1-step10000-tokens42B" "stage1-step15000-tokens63B" "stage1-step20000-tokens84B" "stage1-step50000-tokens210B" "stage1-step110000-tokens462B" "stage1-step928646-tokens3896B")
checkpoint_list=("stage1-step102500-tokens860B" "stage1-step107500-tokens902B" "stage1-step113000-tokens948B" "stage1-step128000-tokens1074B" "stage1-step171000-tokens1435B" "stage1-step221000-tokens1854B" "stage1-step271000-tokens2274B" "stage1-step320000-tokens2685B" "stage1-step596057-tokens5001B")
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