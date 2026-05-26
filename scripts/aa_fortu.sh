#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044"

max_cycles=4
max_layer=18
# model_name="EleutherAI/pythia-70m"
# model_name="EleutherAI/pythia-1.4b"
# model_name="EleutherAI/pythia-6.9b"
# model_name="Qwen/Qwen2.5-7B"
# model_name="google/gemma-2-2b-it"
# model_name="meta-llama/Llama-3.2-1B"
model_name="Qwen/Qwen2.5-1.5B-Instruct"
for n_cyc in $(seq 0 $max_cycles); do
    for n_layer in $(seq 0 $max_layer); do
        JOB_NAME="aa_fortu_${n_cyc}_${n_layer}"
        save_path="/home/mmahaut/projects/parrots/outputs/${model_name}_human_lama_parrots_list_v1_sf/perturbations/aa_fortu/$JOB_NAME"
        mkdir -p $(dirname $save_path)
        ERROR_LOG="$save_path.err"
        OUTPUT_LOG="$save_path.out"
        base_path=$(dirname $(dirname $save_path))
        # mkdir -p $base_path
        script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
which python
cd ~/projects/parrots/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu.py \
    --n-cycles=$n_cyc \
    --single-lens=$n_layer \
    --base-path=$base_path \
    --model-name=$model_name \
    --max-layer-idx=$max_layer 
EOF
)
#         --lens-path="/home/mmahaut/projects/parrots/lens"


        sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES
    done
done
#     --lens-path="/home/mmahaut/projects/parrots/lenses/${model_name}" 
