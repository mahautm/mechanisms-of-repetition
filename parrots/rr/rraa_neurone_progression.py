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

if __name__ == "__main__":
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
    df["value"] = df["value"].astype(int) / df["dataset_size"]
    df.rename(columns={'revision': 'Checkpoint nº'}, inplace=True)

    # take the top 10 most activated heads
    top_heads = df.groupby(['layer', 'head'])['value'].sum().nlargest(10).index
    top_heads = [(layer, head) for layer, head in top_heads]
    print(top_heads)

    df["head id"] = df.apply(lambda x: str(x['layer']) + "." + str(x['head']), axis=1)

    # plot x is idx, y is value, hue is head
    plt.figure(figsize=(10, 6))
    cycle_sizes = df['cycle_size'].unique()
    num_cycles = len(cycle_sizes)
    fig, axes = plt.subplots(num_cycles, 1, figsize=(10, 6 * num_cycles), sharex=True)

    for ax, cycle_size in zip(axes, cycle_sizes):
        sns.lineplot(
            data=df[(df['cycle_size'] == cycle_size) & df[['layer', 'head']].apply(lambda x: (x['layer'], x['head']) in top_heads, axis=1)],
            x='idx', y='value', hue='head id', markers=True, dashes=False, ax=ax
        )
        ax.set_title(f'Top 10 most activated heads - Cycle size {cycle_size}')
        ax.set_xlabel('idx')
        ax.set_ylabel('Number of activations')

    plt.tight_layout()
    plt.savefig(f"{save_folder}/top10_heads_by_cycle_size.png")
