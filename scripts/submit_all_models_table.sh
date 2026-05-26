#!/bin/bash
# Submit a 10-rank array for each model in the candidate list.
# Heuristic: 2 GPUs for models with '7B'|'8B'|'6.9b', otherwise 1 GPU.

models=(
  "EleutherAI/pythia-70m"
  "EleutherAI/pythia-1.4b"
  "EleutherAI/pythia-6.9b"
  "meta-llama/Llama-3.2-1B"
  "allenai/OLMo-1B-hf"
  "swiss-ai/Apertus-8B-2509"
  "allenai/OLMo-2-1124-7B"
)

for model in "${models[@]}"; do
  # sanitize model id for file names
  san=$(echo "$model" | sed 's/[^0-9a-zA-Z_]/_/g')

  # choose resources
  if [[ "$model" == *"7B"* || "$model" == *"8B"* || "$model" == *"6.9b"* ]]; then
    gpus=2
    mem=256G
    time_limit="06:00:00"
  else
    gpus=1
    mem=64G
    time_limit="02:00:00"
  fi

  sbatch_file="scripts/run_rank_${san}.sh"
  cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --job-name=cycle_${san}
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --exclude=node044
#SBATCH --array=0-9
#SBATCH --gres=gpu:${gpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}

source ~/.bashrc
conda activate parr

python compute_cycle_descriptive_stats.py \
  --n_ranks 10 \
  --rank \$SLURM_ARRAY_TASK_ID \
  --max_prompts 1000 \
  --prompt_size 512 \
  --max_new_tokens 1000 \
  --batch_size 8 \
  --models ${model} \
  --allow_downloads \
  --device cuda \
  --device_map auto
EOF

  sbatch_out=$(sbatch "$sbatch_file")
  echo "Submitted $model -> $sbatch_out"
done
