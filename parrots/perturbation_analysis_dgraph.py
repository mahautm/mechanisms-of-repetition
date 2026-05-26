import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import numpy as np
# from parrots.cycle_detection import detect_cycles
from tqdm import tqdm
# from rouge_score import rouge_scorer
# import typer

# Initialize tqdm for pandas
tqdm.pandas()


def collect_data(base_path:str):
    paths = list(Path(base_path).glob(f"*.out"))
    df = pd.DataFrame()
    for p in paths:
        with open(p, "r") as f:
            lines = f.readlines()
        yc, yr, yp, topp, nc, icl = None, None, None, None, None, None
        for l in lines:
            l = l.strip()
            if l.startswith("Proportion of cycles detected in top-p sampling"):
                data = l.split(":")[1].strip()
                yc = float(data)
            if l.startswith("ROUGE-L between greedy and top-p cycles"):
                data = l.split(":")[1].strip()
                yr = float(data)
            # if l.startswith("Proportion of sentence which is made of repetitions"):
            if l.startswith("Proportion of complete cycles detected in top-p sampling"):
                data = l.split(":")[1].strip()
                yp = float(data)/100
            if l.startswith("Top-p"):
                data = l.split(" ")[1].strip()
                topp = float(data)
            if l.startswith("n_cycles"):
                data = l.split(":")[1].strip()
                nc = int(data)
            if l.startswith("ICL"):
                data = l.split(":")[1].strip()
                icl = data=="True"
        if yc is not None and yr is not None and yp is not None and topp is not None and nc is not None and icl is not None:
            df = pd.concat([df, pd.DataFrame([{"topp":topp, "Cycle nº":nc, "icl":icl, "yc":yc, "yr":yr, "yp":yp}])], ignore_index=True)
        else:
            print(f"Missing data in {p}, skipping")
    return df



def plot(df):
    print(df.columns)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # "Proportion of cycles detected in top-p sampling"
    sns.lineplot(ax=axes[0,0], data=df[df["icl"]==False], x="topp", y="yc", hue="Cycle nº", size="Cycle nº", palette="flare")
    axes[0,0].set_ylim(0, 1.03)
    axes[0,0].legend(title="Cycle nº", fontsize=18, title_fontsize=18)
    axes[0,0].set_ylabel("Proportion of Repetitions", fontsize=18)
    axes[0,0].set_xlabel("", fontsize=18)
    axes[0,0].tick_params(axis='both', which='major', labelsize=16)
    axes[0,0].set_xticklabels([])



    sns.lineplot(ax=axes[0,1], data=df[df["icl"]==True], x="topp", y="yc", hue="Cycle nº", size="Cycle nº", palette="flare", legend=False)
    axes[0,1].set_ylim(0, 1.03)
    axes[0,1].set_ylabel("")
    axes[0,1].set_xlabel("", fontsize=18)
    axes[0,1].tick_params(axis='both', which='major', labelsize=16)
    axes[0,1].set_yticklabels([])
    axes[0,1].set_xticklabels([])


    # "ROUGE-L between greedy and top-p cycles"
    sns.lineplot(ax=axes[1, 0], data=df[df["icl"]==False], x="topp", y="yr", hue="Cycle nº", size="Cycle nº", palette="flare", legend=False)
    axes[1,0].set_ylim(0, 1.03)
    axes[1,0].set_ylabel("ROUGE-L Greedy/Top-p Cycles", fontsize=18)
    axes[1,0].set_xlabel("Top-p Sampling Probability", fontsize=18)
    axes[1,0].tick_params(axis='both', which='major', labelsize=16)
    # # "Proportion of sentence which is made of repetitions"
    # sns.lineplot(ax=axes[0], data=df[df["icl"]==True], x="topp", y="yp", hue="Cycle nº", size="Cycle nº", palette="flare")
    # sns.lineplot(ax=axes[0], data=df[df["icl"]==True], x="Cycle nº", y="yp", hue="topp", size="topp", palette="viridis")



    sns.lineplot(ax=axes[1,1], data=df[df["icl"]==True], x="topp", y="yr", hue="Cycle nº", size="Cycle nº", palette="flare", legend=False)
    # sns.lineplot(ax=axes[1], data=df[df["icl"]==False], x="topp", y="yp", hue="Cycle nº", size="Cycle nº", palette="viridis")
    axes[1,1].set_ylim(0, 1.03)
    axes[1,1].set_ylabel("", fontsize=18)
    axes[1,1].set_xlabel("Top-p Sampling Probability", fontsize=18)
    axes[1,1].tick_params(axis='both', which='major', labelsize=16)
    axes[1,1].set_yticklabels([])


    plt.tight_layout()
    plt.show()
    fig.savefig("perturbation_graph_both.png")


if __name__ == "__main__":
    # model_name = "EleutherAI/pythia-1.4b"
    # model_name = "EleutherAI/pythia-70m"
    # model_name = "EleutherAI/pythia-6.9b"
    # model_name = "Qwen/Qwen2.5-7B"
    model_name = "meta-llama/Llama-3.2-1B"

    basepath = f"/home/mmahaut/projects/parrots/outputs/{model_name}_human_lama_parrots_list_v1_sf/perturbations/topp"
    df=collect_data(basepath)
    plot(df)


    