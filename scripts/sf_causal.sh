#!/bin/bash

MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044"


# model="facebook/opt-1.3b"
model="EleutherAI/pythia-1.4B"
base_folder="/home/mmahaut/projects/parrots/outputs/factual/pile_text_per_head" # don't forget to change save folder
# all folders in base_folder
folders=$(ls $base_folder)
for folder in $folders ; do
    mname="${model#*/}"
    JOB_NAME="sf_causal_${mname}_${folder}"
    dir="/home/mmahaut/projects/exps/parr/sf_causal"
    mkdir -p $dir
    ERROR_LOG="$dir/$JOB_NAME.err"
    OUTPUT_LOG="$dir/$JOB_NAME.out"

    script=$(cat <<EOF
source ~/.bashrc
conda activate parr
cd /home/mmahaut/projects/parrots
poetry run python -m parrots.sf_from_causal \
    $base_folder/$folder \
    $model \
    "/home/mmahaut/projects/parrots/outputs/${model}_${folder}_sf_pile" \
    --batch-size 64 \
    --max-new-tokens 100 \
    --log-file /home/mmahaut/projects/parrots/logs/causal_${model}_${folder}_sf_pile.log \
    --use-bnb \
    --no-use-accelerator \

EOF
    )
    sbatch --wrap="$script" --mem=$MEM --partition=$PARTITION --gres=$GRES --qos=$QOS --exclude=$EXCLUDE --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG

done


