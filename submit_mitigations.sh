#!/bin/bash

# Clear empty leftover templates 
rm run_*_p*.sh run_*_icl.sh 2>/dev/null

models=(
    "meta-llama/Llama-3.2-1B Llama"
    "facebook/opt-1.3b OPT"
    "allenai/OLMo-1B-hf OLMo"
    "swiss-ai/Apertus-8B-2509 Apertus"
    "EleutherAI/pythia-1.4b Pythia-1.4B"
)

for item in "${models[@]}"; do
    read -r model name <<< "$item"
    out_dir="outputs/mitigations"
    
    for p in 0.5 0.9; do
        cat << INNER_EOF > run_${name}_p${p}.sh
#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=m_${name}_p${p}
#SBATCH --output=logs/run_${name}_p${p}_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv ${model} ${out_dir}/${name}_p${p} --batch-size 8 --use-bnb --max-new-tokens 512 --top-p ${p} --log-file logs/${name}_p${p}.log
INNER_EOF
        sbatch run_${name}_p${p}.sh
    done

    cat << INNER_EOF > run_${name}_icl.sh
#!/bin/bash
#SBATCH -p alien
#SBATCH -q alien
#SBATCH --exclude=node044
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=m_${name}_icl
#SBATCH --output=logs/run_${name}_icl_%j.out

source ~/.bashrc
conda activate parr
python parrots/slot_filling.py data/human_lama_parrots_list_v1.csv ${model} ${out_dir}/${name}_icl --batch-size 8 --use-bnb --max-new-tokens 512 --use-icl --log-file logs/${name}_icl.log
INNER_EOF
    sbatch run_${name}_icl.sh

done
