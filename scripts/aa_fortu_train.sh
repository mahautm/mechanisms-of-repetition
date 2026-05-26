#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE="node044,node043"
# model_name="Qwen/Qwen2.5-7B"
# model_name="EleutherAI/pythia-70m"
# model_name="meta-llama/Llama-3.2-1B"
model_name=mistralai/Mistral-7B-v0.3
# model_name=google/gemma-2-2b-it
# model_name="EleutherAI/pythia-1.4b"
max_lay_idx=31
for lay_idx in $(seq 0 $max_lay_idx); do
    JOB_NAME="70m_fortu"
    save_path="/home/mmahaut/projects/exps/parr/${model_name}/afortu_${lay_idx}"
    mkdir -p $(dirname $save_path)
    ERROR_LOG="$save_path.err"
    OUTPUT_LOG="$save_path.out"
    script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
which python
cd ~/projects/parrots/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/aa_fortu_train_lens_2.py \
    --layer-idx=$lay_idx \
    --model-name=$model_name 
EOF
)
    sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES
done