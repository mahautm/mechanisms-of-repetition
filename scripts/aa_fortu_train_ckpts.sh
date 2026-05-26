#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:0"
QOS="alien"
EXCLUDE=node044,node043,node042,node041,node040
# model_name="Qwen/Qwen2.5-7B"
# model_name="EleutherAI/pythia-1.4b"
checkpoint_list=("step1" "step1000" "step5000" "step7000" "step10000" "step100000" )
model_name="EleutherAI/pythia-6.9b"
max_layer=32
for lay_idx in $(seq 0 $(($max_layer - 1))); do
    for checkpoint in "${checkpoint_list[@]}"; do
        JOB_NAME="6.9b_fortu"
        save_path="/home/mmahaut/projects/exps/parr/${model_name}/afortu_${lay_idx}"
        mkdir -p $(dirname $save_path)
        ERROR_LOG="$save_path.err"
        OUTPUT_LOG="$save_path.out"
        JOB_NAME="${JOB_NAME}_${lay_idx}_${checkpoint}"
        save_path="${save_path}_${checkpoint}"
        # force download of model before training
        # poetry run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${model_name}', revision='${checkpoint}', force_download=True)"
        script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
which python
cd ~/projects/parrots/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu_train_lens.py \
    --layer-idx=$lay_idx \
    --model-name=$model_name \
    --revision=$checkpoint \
    --do-bfloat16
EOF
)
        sbatch --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES --wrap="$script"
        # sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES
    done
done