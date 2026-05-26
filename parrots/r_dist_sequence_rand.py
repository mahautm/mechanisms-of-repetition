import pandas as pd
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from parrots.rank_distance_sequence import get_logits_rdistance
from parrots.random_rep import data_generation
# Initialize tqdm for pandas
tqdm.pandas()

# Example usage
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = "EleutherAI/pythia-1.4B"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cycle_sizes  = [2,3,5,10]
    n_cycles = [2]
    data_size=1000
    batch_size=1
    df = pd.DataFrame()
    for cycle_size in cycle_sizes:
        for n_cycle in n_cycles:
            cycle_data = data_generation(tokenizer, cycle_size, n_cycle, data_size, batch_size)
            cycle_data = list(cycle_data)
            df = pd.concat([df, pd.DataFrame({"to_send": cycle_data, "cycle_size": cycle_size, "n_cycle": n_cycle})])
    df["generated"]=df["to_send"].progress_apply(lambda x: model.generate(x.to(device), attention_mask=torch.ones_like(x, device=device), max_length=100, pad_token_id=tokenizer.pad_token_id)[0].cpu())
    # make a temp save
    df.to_csv("temp.csv")
    df['rdistance'] = df['generated'].progress_apply(get_logits_rdistance, model=model, tokenizer=tokenizer, device=device)

    for csize in df["cycle_size"].unique():
        graph_df = df[df["cycle_size"] == csize]
        y=[]
        error_bars=[]
        max_entropy_len = max(graph_df["rdistance"].apply(len))
        prompt_size=csize*2
        for i in range(-prompt_size, max_entropy_len):
            # find at which index the cycle begins
            error = np.std(graph_df.apply(lambda x: x["rdistance"][i + prompt_size] if i + prompt_size < len(x["rdistance"]) else None, axis=1).dropna())
            y_i = np.mean(graph_df.apply(lambda x: x["rdistance"][i + prompt_size] if i + prompt_size < len(x["rdistance"]) else None, axis=1).dropna())
            y.append(y_i)
            error_bars.append(error)
        

        plt.plot(range(-prompt_size, max_entropy_len), y, label=f"Cycle size: {csize} (n={len(graph_df)})")
        plt.fill_between(range(-prompt_size, max_entropy_len), np.array(y) - np.array(error_bars), np.array(y) + np.array(error_bars), alpha=0.2)

    # plt.axvline(x=0, color='r', linestyle='--', label="cycle start", alpha=0.5)
    plt.legend()
    plt.xlabel("Token position in sequence")
    plt.ylabel("Absolute rank distance")
    outpath="/home/mmahaut/projects/parrots/outputs/"

    plt.savefig(f"{outpath}/rdistance_rand_rep.png")
    plt.clf()