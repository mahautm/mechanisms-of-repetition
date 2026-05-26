from emecom.dataset_distance import infimb_from_dist
import glob
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

def main():
    # model1_path = "/home/mmahaut/projects/emecom/interm_dist_orig/deit3"
    all_path = "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/dists_icl/"
    all_path2 = "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/dists/"
    # path in dists is dists/layerX/cycleY/BATCH_Z.pickle
    all_paths = sorted(Path(all_path).rglob("*.pickle"))
    max_layer = max([int(p.parts[-3].split("layer_layer_")[1]) for p in all_paths])
    max_cycle = max([int(p.parts[-2].split("cycle_")[1]) for p in all_paths])

    heatmap = np.zeros((max_layer+1, max_cycle+1))
    second_heatmap = np.zeros((max_layer+1, max_cycle+1))
    for i in tqdm(range(max_layer+1), desc="layers"):
        for j in tqdm(range(max_cycle+1), desc=f"Comparing layer {i}", leave=False):
            if heatmap[i,j] != 0:
                continue
            imb1, imb2 = infimb_from_dist(f"{all_path}/layer_layer_{i}/cycle_{j}/*.pickle", f"{all_path2}/layer_layer_{i}/cycle_{j}/*.pickle", k=100)
            heatmap[i,j] = imb1
            second_heatmap[i,j] = imb2
            tqdm.write(f"Layer {i} vs Layer {j}: Imbalance1 = {imb1}, Imbalance2 = {imb2}")
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    sns.heatmap(heatmap, cmap="viridis", cbar_kws={'label': 'Imbalance'}, ax=ax[0])
    ax[0].set_xlabel("Cycle")
    ax[0].set_ylabel("Layer")
    # ax[0].set_title("Information Imbalance cycle0 --> cycleX at each layer")
    ax[0].set_title("Information Imbalance base loop --> icl loop at each layer")

    
    sns.heatmap(second_heatmap, cmap="viridis", cbar_kws={'label': 'Imbalance'}, ax=ax[1])
    ax[1].set_xlabel("Cycle")
    ax[1].set_ylabel("Layer")
    # ax[1].set_title("Information Imbalance cycleX --> cycle0 at each layer")
    ax[1].set_title("Information Imbalance icl loop --> base loop at each layer")
    
    plt.tight_layout()
    plt.savefig(f"{all_path}/imbalance_comparison.png")

if __name__ == "__main__":
    main()   
