import glob
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import warnings
import pandas as pd

def process_file(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        return
    with open(file_path, 'r') as f:
        result={}
        for line in f:
            if 'Score: ' in line:
                parts = line.split('Score')
                if len(parts) > 1:
                    number = parts[1].strip().split()[0]
                    match = re.search(r'\d+', parts[1].strip())
                    if match:
                        number = match.group()
                        result["Portion of cycles"] = number
            if 'random_rep params are:' in line:
                params_line = line.split('random_rep params are:')[1].strip()
                for param in params_line.split(','):
                    key, value = param.split('=')
                    result[key.strip()] = value.strip()
            # if the file ends with nothing, we raise an error
            if not line:
                raise warnings.warn(f"File {file_path} ends with nothing")
        return result

def find_number_after_keyword(base_folder):
    files = glob.glob(f"{base_folder}/**/*.out", recursive=True)
    all_results = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, files)
        for result in results:
            if result:
                all_results.append(result)
    return all_results

# base_folder = '/home/mmahaut/projects/exps/parr/facebook'
# base_folder = '/home/mmahaut/projects/exps/parr/EleutherAI'
base_folder = "/home/mmahaut/projects/exps/parr/allenai"
save_folder = base_folder
all_results = find_number_after_keyword(base_folder)
if not all_results:
    raise ValueError("No results found")
df = pd.DataFrame(all_results)
print(df.head())    
# set the types of the columns
df['revision'] = df['revision'].apply(lambda x: int(re.search(r'\d+', x.split("step")[1]).group()) if x else None)
df = df.astype({'n_cycles': int, 'cycle_size': int, 'dataset_size': int, 'batch_size': int, 'Portion of cycles': int, 'revision': int})
df.rename(columns={'revision': 'Checkpoint nº'}, inplace=True)
# normalize the score
df['Portion of cycles'] = df['Portion of cycles'] / df['dataset_size']

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="colorblind")
for model_name, group in df.groupby('model_name'):
    if len(group['Checkpoint nº'].unique()) == 1:
        fig, ax = plt.subplots()
        sns.lineplot(data=group, x='cycle_size', y='Portion of cycles', hue='n_cycles', ax=ax, palette='colorblind')
        ax.set_title(f"Model {model_name}")

    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(group['cycle_size'].unique()), figsize=(15, 5), sharey=True)
        for ax, (cycle_size, group_cycle) in zip(axes, group.groupby('cycle_size')):
            sns.lineplot(data=group_cycle, x='Checkpoint nº', y='Portion of cycles', hue='n_cycles', ax=ax, palette='colorblind')
            ax.set_title(f"Cycle size {cycle_size}")
        fig.suptitle(f"Model {model_name}")
    file_name = model_name.replace('/', '_') 
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close()
    plt.clf()


