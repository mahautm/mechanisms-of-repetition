# load all jsonfiles in the directory
# get from file_name parameters of the experiment
# plot for each model number of per layer activation
# if there are multiple checkpoints available, they are the x axis

from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import warnings
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def process_file(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        return
    # load json
    with open(file_path, 'r') as f:
        result = json.load(f)
        # if the file ends with nothing, we raise an error
        if not result:
            raise warnings.warn(f"File {file_path} ends with nothing")
        return result

def collect_files(base_folder):
    files = glob.glob(f"{base_folder}/**/hprraa_th0.01/*.json", recursive=True)
    # filter names
    all_results = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, files)
        for result in results:
            if result:
                all_results.append(result)
    return all_results

def group_columns(att_dict):
    _return = []
    _params = {}
    for k, v in att_dict.items():
        if k == "params":
            for kk, vv in v.items():
                _params[kk] = vv
        else:
            n=k.split(".")[0]
            n2=k.split(".")[1]
            _return.append({"layer": n, "head": n2, "value": int(v)})
    
    for r in _return:
        for k, v in _params.items():
            r[k] = v
    return _return
# base_folder = '/home/mmahaut/projects/exps/parr/facebook'
base_folder = '/home/mmahaut/projects/exps/parr/EleutherAI'
save_folder = base_folder
all_results = collect_files(base_folder)
# filter for indexed data
if not all_results:
    raise ValueError("No results found")
print(f"Found {len(all_results)} results")
all_results = [d for d in all_results if "params" in d.keys()]
print(f"Found {len(all_results)} results with params")
all_results = [d for d in all_results if "idx" in d["params"]]
print(f"Found {len(all_results)} results with idx")


all_results = [pd.DataFrame(group_columns(att_dict)) for att_dict in all_results]
df = pd.concat(all_results)
df.dropna(subset=['revision'], inplace=True)

df['revision'] = df['revision'].apply(lambda x: int(re.search(r'\d+', x).group()) if x else None)
df = df.astype({'cycle_size': int, 'dataset_size': int, 'batch_size': int, 'revision': int, 'layer': int, 'idx': int, 'head': int})
df.rename(columns={'revision': 'Checkpoint nº'}, inplace=True)

# filter for highest value of Checkpoint nº
# df = df[df["Checkpoint nº"] == df["Checkpoint nº"].max()]
# df["Number of activations"] = df.apply(lambda x: x["Number of activations"] - df[(df["layer"]==x["layer"]) & (df["idx"] == x["idx"]) & (df["cycle_size"] == 1)]["Number of activations"].sum(), axis=1)
# df = df[df["idx"].isin([0,1,2,3,4])]
# df = df[df["cycle_size"].isin([0,1,2,3,4,5])]
print(df)
# plot for each model


sns.set_theme(style="whitegrid", palette="colorblind")
# print all unique values for all columns
print([f"{col}: {df[col].unique()}" for col in df.columns])
for model_name in df["model_name"].unique():
    model_df = df[df["model_name"] == model_name]
    num_xplots = len(model_df["idx"].unique())
    num_yplots = len(model_df["cycle_size"].unique())
    fig, axes = plt.subplots(num_yplots, num_xplots, figsize=(10 * num_xplots, 10 * num_yplots), sharey=True)
    
    for i, cycle_size in enumerate(sorted(model_df["cycle_size"].unique())):
        cycle_df = model_df[model_df["cycle_size"] == cycle_size]
        for j, _idx in enumerate(sorted(cycle_df["idx"].unique())):
            print(f"Plotting model {model_name} - Cycle Size {cycle_size} - token index {_idx}")
            ax = axes[i][j] if num_yplots > 1 and num_xplots > 1 else axes[max(i, j)]
            cycle_idx_df = cycle_df[cycle_df["idx"] == _idx]
            # find where the duplicates preventing pivot are
            if cycle_idx_df.duplicated(subset=["layer", "head"]).any():
                print(f"Found duplicates in model {model_name} - Cycle Size {cycle_size} - Checkpoint {_idx}")
            cycle_idx_df = cycle_idx_df.drop_duplicates(subset=["layer", "head"])
            # normalize values
            assert cycle_idx_df["dataset_size"].nunique() == 1
            cycle_idx_df["value"] = cycle_idx_df["value"] / cycle_idx_df["dataset_size"].unique()[0]
            cycle_pivot_df = cycle_idx_df.pivot(index="layer", columns="head", values="value")
            sns.heatmap(cycle_pivot_df, ax=ax, cmap="viridis", cbar_kws={'label': 'values'})
            ax.set_title(f"Model {model_name} - Cycle Size {cycle_size} - Token index {_idx}")
            ax.set_xlabel('Head')
            ax.set_ylabel('Layer')
    print(f"Saving model {model_name}")
    plt.tight_layout()
    save_name=model_name.replace("/", "_")
    plt.savefig(f"{save_folder}/rraaidx_{save_name}.png")
    plt.close()
    plt.clf()
