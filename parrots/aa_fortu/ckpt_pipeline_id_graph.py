# the input file looks like this for different layers

# Layer 0: {'blocks.0.attn.hook_result_head_0': array([ 0.25,  0.13,  0.65, 31.09]), 'blocks.0.attn.hook_result_head_1': array([ 0.25,  0.13,  0.64, 25.3 ]), 'blocks.0.attn.hook_result_head_2': array([ 0.25,  0.17,  0.33, 24.9 ]), 'blocks.0.attn.hook_result_head_3': array([ 0.22,  0.5 ,  0.31, 25.89]), 'blocks.0.attn.hook_result_head_4': array([ 0.25,  0.22,  0.67, 24.07]), 'blocks.0.attn.hook_result_head_5': array([ 0.29,  0.47,  0.33, 29.01]), 'blocks.0.attn.hook_result_head_6': array([ 0.2 ,  0.19,  0.68, 23.67]), 'blocks.0.attn.hook_result_head_7': array([ 0.25,  0.2 ,  0.32, 23.51]), 'blocks.0.attn.hook_result_head_8': array([ 0.2 ,  0.24,  0.33, 25.69]), 'blocks.0.attn.hook_result_head_9': array([ 0.22,  0.17,  0.68, 22.79]), 'blocks.0.attn.hook_result_head_10': array([ 0.25,  0.17,  0.33, 22.15]), 'blocks.0.attn.hook_result_head_11': array([ 0.3 ,  0.21,  0.32, 25.02]),

# this script should plot for each layer and head (in order) the 4 values in the array

import pandas as pd
import numpy as np
from numpy import array
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

def load_and_plot_id():
    base_file = "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b/"
    # step{step_number}/ID_{cycle_size}_{t_size}_{range_max}.out
    files=Path(base_file).glob("step*/ID_*.out")

    data = []
    # columns will be {"Layer": int, "Head": int, "Scale": int, "ID": array, "step": int, "cycle_size": int, "t_size": int, "range_max": int}
    for input_file in files:
        with open(input_file, 'r') as f:
            if "latest" in input_file.stem:
                # skip the latest file
                continue
            print(f"Processing file: {input_file}")
            for line in f:
                data_type="Natural"
                if "ICL IDs:" in line:
                    data_type="ICL"
                if "Layer " in line and "blocks." in line:
                    line_data = {
                        "Step": int(input_file.parent.stem.split('step')[1]),
                        "cycle_size": int(input_file.stem.split('_')[1]),
                        "t_size": int(input_file.stem.split('_')[2]),
                        "range_max": int(input_file.stem.split('_')[3]),
                        "Type": data_type
                    }
                    line = line.strip().split(f"Layer ")[1]
                    key = line.split(":")[0].strip()
                    value = [float(x) for x in line.split(":")[1].split("[")[1].split("]")[0].strip().split(" ") if x]

                    # split key into block and head
                    layer = key.split('.')[1]
                    head = key.split('_')[-1]
                    line_data["Layer"] = int(layer)
                    line_data["Head"] = int(head)
                    for i, v in enumerate(value):
                        line_data=line_data.copy()  # make a copy to avoid overwriting
                        line_data["ID"] = v 
                        line_data["Scale"]=2**(i+1)  
                        data.append(line_data)      

    df = pd.DataFrame.from_dict(data)
    # delete duplicate rows
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    # rename the first column to "value"
    df=df[df["Scale"]== 2]  # filter for scale 2
    df=df[df["Type"]=="Natural"]  # filter for natural IDs

    # print(df.head(32))

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    # Create one subplot per layer
    layers = sorted(df["Layer"].unique(), key=lambda x: int(x))
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4 * n_layers), sharex=True)

    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        layer_df = df[df["Layer"] == layer]
        sns.lineplot(data=layer_df, x="Step", y="ID", hue="Head", 
                     style="Head", markers=True, dashes=False, palette="tab10", ax=ax)
        ax.set_title(f"Intrinsic Dimension - Layer {layer}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Intrinsic Dimension")
        ax.legend(title="Head", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("intrinsic_dimension_per_layer.png", dpi=300)
    



if __name__ == "__main__":
    load_and_plot_id()
    