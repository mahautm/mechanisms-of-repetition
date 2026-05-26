#!/bin/bash

# Parameters
MEM="100G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE=node044,node043
# model_name="Qwen/Qwen2.5-7B"
# model_name="EleutherAI/pythia-70m"
# checkpoint_list=("step1" "step1000" "step5000" "step7000" "step10000" "step100000" )
model_name="EleutherAI/pythia-1.4b"
max_layer=1
max_cycles=2
lengths=(32)
# for ckpt in "${checkpoint_list[@]}"; do
#     python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-6.9b', revision='$ckpt', force_download=True)"
# done
for max_length in "${lengths[@]}"; do
    # for n_cyc in $(seq 0 $max_cycles); do
    for n_cyc in 2; do
        for lay_idx in $(seq 0 $(($max_layer - 1))); do
            # for checkpoint in "${checkpoint_list[@]}"; do
            checkpoint=step0
            echo "Running for n_cyc=$n_cyc, lay_idx=$lay_idx, checkpoint=$checkpoint"
            JOB_NAME="pythia-1.4b_${n_cyc}_${lay_idx}_${max_length}"
            save_path="/home/mmahaut/projects/parrots/outputs/${model_name}/${checkpoint}/$JOB_NAME"
            mkdir -p $(dirname $save_path)
            ERROR_LOG="$save_path.err"
            OUTPUT_LOG="$save_path.out"
            JOB_NAME="${JOB_NAME}_${lay_idx}_${checkpoint}"
            # force download of model before training
            # poetry run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${model_name}', revision='${checkpoint}', force_download=True)"
            script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
module load CUDA/12.2.0
which python
cd ~/projects/parrots/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/ckpt_pipeline_main.py \
    --n-cycles=$n_cyc \
    --single-lens=$lay_idx \
    --model-name=$model_name \
    --max-layer-idx=$max_layer \
    --lens-path="/home/mmahaut/projects/parrots/lenses_multihead/${model_name}/"  \
    --max-length=$max_length \
    --max-new-tokens=1000 \
    --batch-size=10 \
    --n-samples=100 \
    --revision=step0
EOF
)
    # --revision=$checkpoint \
    # --lens-path="/home/mmahaut/projects/parrots/lenses/${model_name}_${checkpoint}/"  \


                sbatch --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES --wrap="$script"
                # sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES
            done
        # done
    done
done

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu.py \
# --base-path=$base_path \
# --sample=300
# --use-bfloat16 


