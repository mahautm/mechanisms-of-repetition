import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model_names = ["EleutherAI/pythia-70m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-6.9b"]
    # model_names=["Qwen/Qwen2.5-7B"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, model_name in zip(axes, model_names):
        base_path = f"/home/mmahaut/projects/parrots/outputs/{model_name}_human_lama_parrots_list_v1_sf/perturbations"
        df = pd.DataFrame()
        for f in Path(base_path).glob(f"cycle_{3}_results_*.csv"):
            df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
        print(f"{model_name} proportion of cycles detected in data:", len(df[df["cycle_size"]!=0])/len(df))
        # df = df[df["cycle_size"] != 0]
        # df = df[df["cycle_size"] != 1]
        df = df[df["cycle_count"] != 0]
        df = df[df["cycle_count"] != 1]
        df["cycle_size"] = df["cycle_size"].replace([np.inf, -np.inf], np.nan).dropna()
        df = df[["cycle_size"]]

        sns.histplot(data=df, x="cycle_size", stat='probability', ax=ax)
        ax.set_title(model_name, fontsize=18)
        ax.set_ylabel("Proportion of Dataset", fontsize=18)
        ax.set_xlabel("Cycle size", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig("/home/mmahaut/projects/parrots/outputs/cycle_size_distribution_comparison.png")
    plt.show()