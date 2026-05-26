#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044"
NICE=12

rank_numbers=($(seq 200 400))
n_ranks=20000
batch_size=1
# model_name=Qwen/Qwen2.5-7B
# first, download model once from huggingface
# poetry run python -m transformers.AutoModel.from_pretrained $model_name

# model_name="mistralai/Mistral-7B-v0.3"
# model_name="EleutherAI/pythia-6.9b"
# model_name="EleutherAI/pythia-1.4b"
# model_name="EleutherAI/pythia-70m"
# model_name="meta-llama/Llama-3.2-1B"
# model_name="google/gemma-2-2b-it"
# model_name="meta-llama/Llama-3.2-1B"
# model_name="allenai/OLMo-2-0425-1B-Instruct"
model_name="Qwen/Qwen2.5-1.5B-Instruct"

max_new_tokens=200
cycle_sizes=(3)

for cycle_size in "${cycle_sizes[@]}" 
do
    for rank_number in "${rank_numbers[@]}"
    do
        echo "Rank number: $rank_number"
        echo "Cycle size: $cycle_size"
        JOB_NAME="per_${rank_number}_${cycle_size}_$(echo $model_name | tr '/' '_')"
        save_path="/home/mmahaut/projects/parrots/outputs/${model_name}_human_lama_parrots_list_v1_sf/perturbations"
        # save_path=/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations
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
poetry run python /home/mmahaut/projects/parrots/parrots/pile_perturbations.py \
    --rank-number=$rank_number \
    --n-ranks=$n_ranks \
    --max-new-tokens=$max_new_tokens \
    --cycle-size=$cycle_size \
    --batch-size=$batch_size \
    --model-name=$model_name \
    --save-path=$save_path \
    --skip-phase-2
EOF
        )

        sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --gres=$GRES --qos=$QOS --exclude=$EXCLUDE --nice=$NICE
    done
done