
all_layer_path=/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/hlayers_icl/.pkl
# dirs have format layer0, layer1, layer2, etc.
for layer_dir in /home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/hlayers_icl/layer*
do
    layer=$(basename $layer_dir)
    echo "Processing layer $layer"
    # find highest cycle
    max_cycle=$(ls -1 $layer_dir | grep -o "cycle_[0-9]*" | grep -o "[0-9]*" | sort -n | tail -n 1)
    echo "Max cycle: $max_cycle"
    for cycle in $(seq 0 $max_cycle)
    do
        echo "Processing cycle $cycle"
        poetry run python /home/mmahaut/projects/emecom/emecom/precompute_distances.py \
            "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/hlayers_icl/$layer/*cycle_$cycle*.pkl" \
            /home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/dists_icl/layer_$layer/cycle_$cycle
    done
done