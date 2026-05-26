# Description: This script is used to launch multiple jobs on the cluster using SLURM.
# It is used to launch the script `sf.sh` with different arguments.
# List of models
# models=("mistralai/Mistral-7B-Instruct-v0.3" \
#     "mistralai/Mistral-7B-v0.3" \
#     "facebook/opt-1.3b" \
#     "EleutherAI/pythia-1.4b" \
#     "EleutherAI/pythia-6.9b" \
#     "allenai/OLMo-1B-hf" \
#     "allenai/OLMo-1.7-7B-hf")
models=("EleutherAI/pythia-1.4b")
# datasets=("/home/mmahaut/projects/parrots/data/human_lama_parrots_list_v1.csv" \
#          "/home/mmahaut/projects/parrots/data/lama.csv" \
#          "/home/mmahaut/projects/parrots/data/autoprompts_opt1_3b_lama_parrot_list_v1.csv")
datasets=("/home/mmahaut/projects/parrots/data/human_lama_parrots_list_v1.csv"
         "/home/mmahaut/projects/parrots/data/autoprompts_opt1_3b_lama_parrot_list_v1.csv")
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        model_name=$(echo $model | cut -d'/' -f2)
        dataset_name=$(echo $dataset | cut -d'/' -f7 | cut -d'.' -f1)
        # Set the output and error file paths
        output_file="/home/mmahaut/projects/exps/parr/sf_${model_name}_${dataset_name}.out"
        error_file="/home/mmahaut/projects/exps/parr/sf_${model_name}_${dataset_name}.err"
        job_name="sf_${model_name}_${dataset}"
        partition="alien"
        qos="alien"
        nodes=1
        exclude="node044"
        nice=12
        ntasks_per_node=1
        time="3-00:00:00"
        mem="100G"

        # Launch the job
        sbatch --output="$output_file" --error="$error_file" --job-name="$job_name" --gres="gpu:1"\
        --partition="$partition" --qos="$qos" --nodes="$nodes" --exclude="$exclude" \
        --nice="$nice" --ntasks-per-node="$ntasks_per_node" --time="$time" --mem="$mem" \
        /home/mmahaut/projects/parrots/scripts/sf.sh $model $dataset
    done
done