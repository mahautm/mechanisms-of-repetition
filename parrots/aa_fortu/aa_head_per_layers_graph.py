from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import typer
from plotly.colors import sample_colorscale
import joypy

def plot_from_proba(base_path, heatmap_type="natural", cycle_size=0, t_size=0):
    # collect logs in base_path
    logs = []
    for f in Path(base_path).glob(f"*_{cycle_size}_*_{t_size}.out"):
        with open(f, "r") as file:
            logs.append(file.read())
    # Extract heatmap elements from logs
    heatmaps = []
    for log in logs:
        # layer 23 natural heatmap: 
        matches = re.findall(fr"layer (\d+) {heatmap_type} heatmap: (.+)", log)
        for layer, match in matches:
            if match == "None":
                heatmap = None
            else:
                heatmap = np.array(eval(match))
            heatmaps.append((int(layer), heatmap))
    heatmaps.sort(key=lambda x: x[0])
    heatmaps = [heatmap for _, heatmap in heatmaps]
    return heatmaps

    

def heads_across_time():
    steps = [1, 1000, 5000, 7000, 10000, 100000]
    steps = [f"step{i}" for i in steps]
    cycle_count = list(range(0, 5))
    t_size=[16,32,64,128,256,512]
    # layers = list(range(0, 24))
    model_name = "EleutherAI/pythia-1.4b"
    proportions = {"cycle_size": [], "t_size": [], "step": [], "contrasts": []}
    # proportion plot 
    for cycle_size in cycle_count:
        for t in t_size:
            for step in steps:
                path = f"/home/mmahaut/projects/parrots/outputs/{model_name}/{step}/"
                contrasts = plot_from_proba(path, cycle_size=cycle_size, t_size=t) # n_layers * n_attention_heads
                if np.shape(contrasts) == (0,):
                    continue
                proportions["cycle_size"].append(cycle_size)
                proportions["t_size"].append(t)
                proportions["step"].append(step)
                proportions["contrasts"].append(contrasts)
    # Convert proportions to DataFrame for easier plotting
    df = pd.DataFrame({
        "cycle_size": proportions["cycle_size"],
        "t_size": proportions["t_size"],
        "step": proportions["step"],
        "contrasts": proportions["contrasts"]
    })

    # Expand contrasts into separate rows for each layer
    df["layer"] = df["contrasts"].apply(lambda x: None if x is None else np.arange(len(x)))
    df["head"] = df["contrasts"].apply(lambda x: None if x is None else np.arange(len(x[0])) if len(x) > 0 else None)
    df = df.explode("head")
    df = df.explode("layer")
    df["contrast_value"] = df.apply(lambda row: None if row["contrasts"] is None or row["layer"] is None else row["contrasts"][row["layer"]], axis=1)
    df = df.drop(columns=["contrasts"])
    df["contrast_value"] = df.apply(lambda row: None if row["contrast_value"] is None or row["head"] is None else row["contrast_value"][row["head"]], axis=1)
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)

    # Pivot the DataFrame so each step is a column, indexed by cycle_size, t_size, layer
    pivot_df = df.pivot_table(
        index=["cycle_size", "t_size", "layer", "head"],
        columns="step",
        values="contrast_value"
    ).reset_index()
    # clip all values to be between 0 and 1 fir steps NOT for cycle_size or t_size or layer or head
    for step in steps:
        if step in pivot_df.columns:
            pivot_df[step] = pivot_df[step].clip(0, 1)
    # # delete values too close to 0 by epsilon = 1e-4
    # epsilon = 1e-4
    # for step in steps:
    #     if step in pivot_df.columns:
    #         pivot_df[step] = pivot_df[step].apply(lambda x: x if abs(x) > epsilon else 0)
    # # drop 0 values
    # pivot_df = pivot_df[(pivot_df[steps] != 0).any(axis=1)]
    # print(f"min step values: {pivot_df[steps].min()}, max step values: {pivot_df[steps].max()}")

    # Now, for each t_size and cycle_size, plot boxplots for all steps, grouped by layer
    for t in pivot_df["t_size"].unique():
        for c in pivot_df["cycle_size"].unique():
            subset = pivot_df[(pivot_df["t_size"] == t) & (pivot_df["cycle_size"] == c)]
            if subset.empty:
                print(f"No data for t_size={t}, cycle_size={c}. Skipping boxplot.")
                continue
            # Melt the DataFrame to long format for seaborn boxplot
            melted = subset.melt(
                id_vars=["layer", "head"],
                value_vars=steps,
                var_name="step",
                value_name="contrast_value"
            )
            plt.figure(figsize=(16, 8))
            sns.boxplot(
                data=melted,
                x="layer",
                y="contrast_value",
                hue="step",
                showfliers=False
            )
            plt.title(f"Boxplot of Contrast Value by Layer (t_size={t}, cycle_size={c})")
            plt.xlabel("Layer")
            plt.ylabel("Contrast Value")
            plt.legend(title="Step", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"./boxplot_contrast_layer_tsize_{t}_cycle_{c}.png")
            plt.close()
            print(f"Saved boxplot for t_size={t}, cycle_size={c} to ./boxplot_contrast_layer_tsize_{t}_cycle_{c}.png")
    


if __name__ == "__main__":
    heads_across_time()

