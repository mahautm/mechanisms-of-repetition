#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044"
NICE=12

# model_name="mistralai/Mistral-7B-v0.3"
# model_name="Qwen/Qwen2.5-7B"
model_name="meta-llama/Llama-3.2-1B"
# model_name="EleutherAI/pythia-6.9b"
# model_name="EleutherAI/pythia-70m"

cycle_sizes=(0 1 2 3 4 5 6)
# topps=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
topps=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for cycle_size in "${cycle_sizes[@]}" 
do
    for topp in "${topps[@]}"
    do
        echo "Top p: $topp"
        echo "Cycle size: $cycle_size"
        JOB_NAME="peran_icl_${topp}_${cycle_size}_$(echo $model_name | tr '/' '_')"
        save_path="/home/mmahaut/projects/parrots/outputs/${model_name}_human_lama_parrots_list_v1_sf/perturbations/topp"
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
poetry run python /home/mmahaut/projects/parrots/parrots/perturbation_analysis_distributed.py \
    $topp \
    --n-cycles=$cycle_size \
    --model-name=$model_name \
    --icl
EOF
        )
        sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --gres=$GRES --qos=$QOS --exclude=$EXCLUDE --nice=$NICE
    done
done