from pathlib import Path
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main(): 
    icl_heatmaps = []
    unexpected_heatmaps = []
    # model_name = "Qwen/Qwen2.5-7B"
    # model_name = "EleutherAI/pythia-70m"
    # step = "step100000"
    model_name = "EleutherAI/pythia-1.4b"
    # model_name = "google/gemma-2-2b-it"
    # model_name="meta-llama/Llama-3.2-1B"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # model_name = "EleutherAI/pythia-6.9b"
    # base_path = f"/home/mmahaut/projects/parrots/outputs/{model_name}/{step}/"
    base_path = f"/home/mmahaut/projects/parrots/outputs/{model_name}_human_lama_parrots_list_v1_sf/perturbations/aa_fortu/"
    print(f"Base path: {base_path}")
    # are there logs in the base path?
    if not Path(base_path).exists():
        print(f"Base path {base_path} does not exist. Exiting.")
        return
    if not any(Path(base_path).glob("*.out")):
        print(f"No logs found in base path {base_path}. Exiting.")
        return

    t_size = 64

    for n_cycles in range(1,5):
        print(f"Cycle number: {n_cycles}")
        full_heatmap = plot_from_logs(base_path, heatmap_type="icl", cycle_size=n_cycles, t_size=t_size) 
        icl_heatmaps.append(full_heatmap)
        # max_layer_idx = 24
        cy_count = 1
        # Plot and save heatmap
        plt.figure(figsize=(10, 8))
        print(full_heatmap)
        if all([h is None for h in full_heatmap]):
            print(f"Heatmap is empty for cycle {n_cycles}")
            continue
        sns.heatmap(full_heatmap, cmap="magma", cbar=True, vmin=-0.12, vmax=0.5)
        plt.title(f"ICL contrast contrast Heatmap")
        plt.suptitle(f"proportion of cycles: {cy_count}, cycle number: {n_cycles}")
        plt.xlabel("Attention Heads")
        plt.ylabel("Samples")
        plt.savefig(f"icl_lens_heatmap_contrast_{n_cycles}.png")
        plt.close()

        full_heatmap = plot_from_logs(base_path, heatmap_type="natural", cycle_size=n_cycles, t_size=t_size) 

        unexpected_heatmaps.append(full_heatmap)
        plt.figure(figsize=(10, 8))
        sns.heatmap(full_heatmap, cmap="magma", cbar=True, vmin=-0.12, vmax=0.5)
        plt.title(f"unexpected contrast contrast Heatmap {n_cycles}")
        plt.suptitle(f"proportion of cycles: {cy_count}, cycle number: {n_cycles}")
        plt.xlabel("Attention Heads")
        plt.ylabel("Samples")
        plt.savefig(f"unexpected_lens_heatmap_contrast_{n_cycles}.png")
        plt.close()

    
    # plt.figure(figsize=(8, 6))
    data1=pd.DataFrame(columns=["head", "contrast", "cycle", "layer"])
    for cycle, icl_heatmap in enumerate(icl_heatmaps):
        for layer, heatmap in enumerate(icl_heatmap):
            for head, contrasts in enumerate(heatmap):
                data1 = pd.concat([data1, pd.DataFrame([{"head": head, "contrast": contrasts, "cycle": cycle, "layer": layer}])], ignore_index=True)
    data1["Layer.Head"] = data1["layer"].astype(int).astype(str) + "." + data1["head"].astype(int).astype(str)
    # keep top 3 head_layers with highest and top 3 with lowest contrasts
    print(data1)
    top2_highest1 = data1.groupby("Layer.Head")["contrast"].mean().nlargest(2).index
    top2_lowest1 = data1.groupby("Layer.Head")["contrast"].mean().nsmallest(2).index

    # also get contrast for heads 4.4 and 9,9
    # forced_layer_heads = ["4.4", "9.9"]
    # top3_highest = top3_highest.union(forced_layer_heads)
    # pdata1 = data1[data1["Layer.Head"].isin(top3_highest.union(top3_lowest))]
    # sort head_layers
    # pdata1 = pdata1.sort_values("Layer.Head")
    # pdata1.columns = pdata1.columns.str.capitalize()
    # pdata1.rename(columns={"Layer.head": "Layer.Head"}, inplace=True)
    # sns.lineplot(data=pdata1, x="Cycle", y="contrast", hue="Layer.Head", style="Layer.Head", markers=True, dashes=False, markersize=10, linewidth=2)
    # additionally plot all data, transparently, with no legend
    # sns.lineplot(data=data1, x="cycle", y="contrast", alpha=0.1, hue="Layer.Head", palette=["grey"] * len(data1["Layer.Head"].unique()), legend=False)
    # plt.ylim(-0.12, 0.25)
    # plt.legend(title="Layer.Head", loc='upper right')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # legend_dict = dict(zip(labels, handles))
    
    # plt.tight_layout()
    # plt.savefig("icl_lens_heatmap_contrast.png")
    
    # plt.close()

    # plt.figure(figsize=(8, 6))
    data=pd.DataFrame(columns=["head", "contrast", "cycle", "layer"])
    for cycle, unexpected_heatmap in enumerate(unexpected_heatmaps):
        for layer, heatmap in enumerate(unexpected_heatmap):
            for head, contrasts in enumerate(heatmap):
                data = pd.concat([data, pd.DataFrame([{"head": head, "contrast": contrasts, "cycle": cycle, "layer": layer}])], ignore_index=True)
    data["Layer.Head"] = data["layer"].astype(int).astype(str) + "." + data["head"].astype(int).astype(str)
    # keep top 3 head_layers with highest and top 3 with lowest contrasts
    top2_highest = data.groupby("Layer.Head")["contrast"].mean().nlargest(2).index
    top2_lowest = data.groupby("Layer.Head")["contrast"].mean().nsmallest(2).index
    # also get contrast for heads 4.4 and 9,9
    # forced_layer_heads = ["4.4", "9.9"]
    # top3_highest = top3_highest.union(forced_layer_heads)
    pdata = data[data["Layer.Head"].isin(top2_highest.union(top2_lowest).union(top2_highest1).union(top2_lowest1))]
    pdata1 = data1[data1["Layer.Head"].isin(top2_highest.union(top2_lowest).union(top2_highest1).union(top2_lowest1))]
    # sort head_layers
    pdata["Layer.Head"] = pdata["Layer.Head"].apply(lambda x: tuple(map(int, x.split('.'))))
    pdata = pdata.sort_values("Layer.Head")
    pdata["Layer.Head"] = pdata["Layer.Head"].apply(lambda x: f"{x[0]}.{x[1]}")
    pdata.columns = pdata.columns.str.capitalize()
    pdata.rename(columns={"Layer.head": "Layer.Head"}, inplace=True)

    pdata1["Layer.Head"] = pdata1["Layer.Head"].apply(lambda x: tuple(map(int, x.split('.'))))
    pdata1 = pdata1.sort_values("Layer.Head")
    pdata1["Layer.Head"] = pdata1["Layer.Head"].apply(lambda x: f"{x[0]}.{x[1]}")    
    pdata1.columns = pdata1.columns.str.capitalize()
    pdata1.rename(columns={"Layer.head": "Layer.Head"}, inplace=True)
    # Use the colors from legend_dict when Layer.Head matches
    # palette = {key: legend_dict[key].get_color() for key in pdata["Layer.Head"].unique() if key in legend_dict}
    # Use the same markers
    # markers = {key: legend_dict[key].get_marker() for key in pdata["Layer.Head"].unique() if key in legend_dict}
    # add missing keys to palette avoiding colours in legend_dict
    # for key in pdata["Layer.Head"].unique():
    #     if key not in palette:
    #         used_colors = set(palette.values()).union([ld.get_color() for ld in legend_dict.values()])
    #         available_colors = [c for c in plt.cm.tab20.colors if c not in used_colors]
    #         c_idx = np.random.choice(range(len(available_colors)))
    #         palette[key] = available_colors[c_idx]
    #     if key not in markers:
    #         used_markers = set(markers.values())
    #         available_markers = [m for m in ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*'] if m not in used_markers]
    #         markers[key] = np.random.choice(available_markers)
    # sns.lineplot(data=pdata, x="Cycle", y="contrast", hue="Layer.Head", style="Layer.Head", markers=markers, dashes=False, palette=palette, markersize=10, linewidth=2)
    # additionally plot all data, transparently, with no legend
    # sns.lineplot(data=data, x="cycle", y="contrast", alpha=0.1, hue="Layer.Head", palette=["grey"] * len(data["Layer.Head"].unique()), legend=False)
    # plt.ylim(-0.12, 0.25)
    # plt.title("unexpected contrast contrast")
    # plt.tight_layout()
    # plt.savefig("unexpected_lens_heatmap_contrast.png")
    # plt.close()

    # Create subplots for both plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))


    sns.lineplot(data=data1, x="cycle", y="contrast", alpha=0.1, hue="Layer.Head", palette=["grey"] * len(data["Layer.Head"].unique()), legend=False, ax=axes[1])
    sns.lineplot(data=data, x="cycle", y="contrast", alpha=0.1, hue="Layer.Head", palette=["grey"] * len(data["Layer.Head"].unique()), legend=False, ax=axes[0])
    # Plot ICL heatmap contrast
    sns.lineplot(data=pdata1, x="Cycle", y="Contrast", hue="Layer.Head", style="Layer.Head", markers=True, dashes=False, markersize=15, linewidth=3, ax=axes[1])
    # Plot unexpected heatmap contrast
    sns.lineplot(data=pdata, x="Cycle", y="Contrast", hue="Layer.Head", style="Layer.Head", markers=True, dashes=False, markersize=15, linewidth=3, ax=axes[0])

    axes[0].set_xlabel("Cycle nº", fontsize=14)
    axes[1].set_xlabel("Cycle nº", fontsize=14)
    axes[0].set_ylabel("Contrast", fontsize=14)
    axes[1].set_ylabel('')
    # Combine legends
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    combined_handles = handles + handles2
    combined_labels = labels + labels2
    unique_labels_handles = dict(zip(combined_labels, combined_handles))
    
    fig.legend(unique_labels_handles.values(), unique_labels_handles.keys(), loc='lower center', bbox_to_anchor=(0.2, 0.12), ncol=4, title="Layer.Head", fontsize=12, title_fontsize=14)
    # deactive legend for individual plots
    axes[0].legend().set_visible(False)
    axes[1].legend().set_visible(False)
    
    # axes[0].set_ylim(-0.2, 0.55)
    # axes[1].set_ylim(-0.2, 0.55)
    axes[1].sharey(axes[0])
    axes[1].set_ylabel('')
    axes[1].set_yticklabels([])
    # sns.lineplot(data=data, x="cycle", y="contrast", alpha=0.1, hue="Layer.Head", palette=["grey"] * len(data["Layer.Head"].unique()), legend=False, ax=axes[1])
    
    # axes[1].set_ylim(-0.2, 0.26)
    # axes[1].set_ylabel('')
    # axes[1].set_yticklabels([])
    # axes[1].legend(title="Layer.Head", loc='upper right', fontsize=12, title_fontsize=14)


    for ax in axes:
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(base_path + f"lens_heatmap_contrast_{t_size}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved lens_heatmap_contrast.png as {base_path + f'lens_heatmap_contrast_{t_size}.png'}")
        

def plot_from_logs(base_path, heatmap_type="natural", cycle_size=0, t_size=0):
    # collect logs in base_path
    logs = []
    # for f in Path(base_path).glob(f"*_{cycle_size}_*_{t_size}.out"):
    for f in Path(base_path).glob(f"*_{cycle_size}_*.out"):
        with open(f, "r") as file:
            logs.append(file.read())
    # Extract heatmap elements from logs
    heatmaps = []
    for log in logs:
        # layer 23 natural heatmap: 
        matches = re.findall(fr"layer (\d+) {heatmap_type} heatmap: (.+)", log)
        for layer, match in matches:
            print(f"Layer: {layer}, Match: {match}")
            if match == "None":
                heatmap = None
            else:
                heatmap = np.array(eval(match))
            heatmaps.append((int(layer), heatmap))
    heatmaps.sort(key=lambda x: x[0])
    heatmaps = [heatmap for _, heatmap in heatmaps]
    return heatmaps

if __name__ == "__main__":
    main()