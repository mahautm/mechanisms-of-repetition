#!/bin/bash
models=(
    "meta-llama/Llama-3.2-1B outputs/meta-llama/Llama-3.2-1B_512_sf Llama"
    "facebook/opt-1.3b outputs/facebook/opt-1.3b_512_sf OPT"
    "allenai/OLMo-1B-hf outputs/allenai/OLMo-1B-hf_512_sf OLMo"
    "swiss-ai/Apertus-8B-2509 outputs/swiss-ai/Apertus-8B-2509_512_sf Apertus"
)
for item in "${models[@]}"; do
    read -r model out_path name <<< "$item"
    cat << INNER_EOF > run_${name}_512.sh
#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=sf512_${name}
#SBATCH --output=logs/run_${name}_512_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv ${model} ${out_path} --batch-size 8 --use-bnb --max-new-tokens 512 --log-file logs/${name}_512.log
INNER_EOF
    sbatch run_${name}_512.sh
done
