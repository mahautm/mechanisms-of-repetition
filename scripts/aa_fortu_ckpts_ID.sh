#!/bin/bash

# Parameters
MEM="250G"
PARTITION="alien"
GRES="gpu:1"
QOS="alien"
EXCLUDE=node044,node043
# model_name="Qwen/Qwen2.5-7B"
# model_name="EleutherAI/pythia-70m"
checkpoint_list=("step1" "step1000" "step5000" "step7000" "step10000" "step100000" )
model_name="EleutherAI/pythia-1.4b"
max_layer=24
n_cyc=2
lengths=(16)
tab_max_n_tokens=(512)
lay_idx=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
# for ckpt in "${checkpoint_list[@]}"; do
#     python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-6.9b', revision='$ckpt', force_download=True)"
# done
for max_length in "${lengths[@]}"; do
    for checkpoint in "${checkpoint_list[@]}"; do
        for max_n_tokens in "${tab_max_n_tokens[@]}"; do
            echo "Running for n_cyc=$n_cyc, lay_idx=$lay_idx, checkpoint=$checkpoint, max_length=$max_length, max_n_tokens=$max_n_tokens"
        JOB_NAME="ID_${n_cyc}_${max_length}_${max_n_tokens}"
        save_path="/home/mmahaut/projects/parrots/outputs/${model_name}/${checkpoint}/$JOB_NAME"
        base_path="/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations"
        mkdir -p $(dirname $save_path)
        ERROR_LOG="$save_path.err"
        OUTPUT_LOG="$save_path.out"
        JOB_NAME="${JOB_NAME}_${checkpoint}"
        # force download of model before training
        # poetry run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${model_name}', revision='${checkpoint}', force_download=True)"
        layer_idx_str=$(for idx in "${lay_idx[@]}"; do echo -n "    --layer-idx=$idx "; done)
        script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate parr
which python
cd ~/projects/parrots/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/ckpt_pipeline_id.py \
    --n-cycles=$n_cyc \
    $layer_idx_str \
    --model-name=$model_name \
    --revision=$checkpoint \
    --max-layer-idx=$max_layer \
    --max-length=$max_length \
    --max-new-tokens=$max_n_tokens \
    --batch-size=16
    
EOF
)
            sbatch --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES --wrap="$script"
            # sbatch --wrap="$script" --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=$MEM --partition=$PARTITION --qos=$QOS --exclude=$EXCLUDE --nice=10 --gres=$GRES
        done
    done
done

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu.py \
# --base-path=$base_path \
# --sample=300
# --use-bfloat16 


