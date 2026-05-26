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
    files = glob.glob(f"{base_folder}/**/hrraa_th0.01/*.json", recursive=True)
    # ignore file that has "average" in the name
    files = [f for f in files if "average" not in f]
    # filter names
    all_results = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, files)
        for result in results:
            if result:
                all_results.append(result)
    return all_results

def group_columns(att_dict):
    _return = {}
    for k, v in att_dict.items():
        if k == "params":
            for kk, vv in v.items():
                _return[kk] = vv
        else:
            n=k.split(".")[0]
            if n not in _return:
                _return[n] = v
            else:
                _return[n] = _return[n] + v
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


all_results = [group_columns(att_dict) for att_dict in all_results]
df = pd.DataFrame(all_results)
df.dropna(subset=['revision'], inplace=True)

df['revision'] = df['revision'].apply(lambda x: int(re.search(r'\d+', x).group()) if x else None)
df = df.astype({'n_cycles': int, 'cycle_size': int, 'dataset_size': int, 'batch_size': int, 'revision': int})
df.rename(columns={'revision': 'Checkpoint nº'}, inplace=True)
print(df.columns)

df = df.melt(id_vars=["n_cycles", "cycle_size", "dataset_size", "batch_size", "model_name", "Checkpoint nº", "idx"], value_vars=[str(i) for i in range(0, 50) if str(i) in df.columns], var_name="layer", value_name="Number of activations")
# filter for highest value of Checkpoint nº
# df = df[df["Checkpoint nº"] == df["Checkpoint nº"].max()]
df["Number of activations"] = df.apply(lambda x: x["Number of activations"] - df[(df["layer"]==x["layer"]) & (df["idx"] == x["idx"]) & (df["cycle_size"] == 1)]["Number of activations"].sum(), axis=1)
df = df[df["cycle_size"] != 1]
print(df)
# plot for each model
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="colorblind")
print(df["model_name"].unique(), df["cycle_size"].unique())
for model_name in df["model_name"].unique():
    model_df = df[df["model_name"] == model_name]
    unique_cycle_sizes = model_df["cycle_size"].unique()
    num_subplots = len(unique_cycle_sizes)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(20, 10 * num_subplots), sharex=True)
    
    if num_subplots == 1:
        axes = [axes]
    
    for ax, cycle_size in zip(axes, unique_cycle_sizes):
        cycle_df = model_df[model_df["cycle_size"] == cycle_size]
        # make n new rows for each Checkpoint nº
        # normalize number of activations to 100
        cycle_df["Number of activations"] =(cycle_df["Number of activations"] / cycle_df["Number of activations"].max() * 100).astype(int)
        cycle_df["Number of activations"]=cycle_df["Number of activations"].apply(lambda x:[1]*x if x > 0 else [])
        cycle_df = cycle_df.explode("Number of activations")
        cycle_df = cycle_df.dropna(subset=["Number of activations"])
        sns.swarmplot(data=cycle_df, x="idx", y="layer", hue="Checkpoint nº", ax=ax, palette="viridis", size=0.5)
        ax.set_title(f"Model {model_name} - Cycle Size {cycle_size}")
    print(f"Saving model {model_name}")
    plt.tight_layout()
    save_name=model_name.replace("/", "_")
    plt.savefig(f"{save_folder}/rraaidx_{save_name}.png")
    plt.close()
    plt.clf()
