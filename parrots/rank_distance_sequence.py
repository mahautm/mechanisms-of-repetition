import pandas as pd
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
# Initialize tqdm for pandas
tqdm.pandas()

def get_logits_rdistance(text, model, tokenizer, device):
    """Extract density for each sequence in the DataFrame."""
    # TODO: don't use tokenizer here, only take tokenized input
    if isinstance(text, str):
        toked = tokenizer(text)
    else:
        warnings.warn(f"input type is {type(text)}, not str, we're assuming it's already tokenized")
        toked = {"input_ids": text.unsqueeze(0)}
    _t = {}
    for k, v in toked.items():
        if isinstance(v, list):
            _t[k] = torch.tensor(v).to(device)
        else:
            _t[k] = v.to(device)
    o = model(**_t)
    all_logs = o.logits[0, :, :].cpu().detach().numpy()
    # probability distance between first two tokens
    rank_distances = []
    for i in range(1, len(all_logs)):
        rank_distance = all_logs[i, :].argsort()[-1] - all_logs[i, :].argsort()[-2]
        rank_distance = abs(rank_distance)
        rank_distances.append(rank_distance)
    return rank_distances

# Example usage

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results_with_cycles.csv")
    # df=pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results_with_cycles.csv")
    df = df[df["cycle"] != "  "]
    df = df[df["cycle_size"] <= 3]
    # sample 50 rows from each cycle size
    df = df.groupby("cycle_size").apply(lambda x: x.sample(min(100, len(x)))).reset_index(drop=True)
    # Initialize the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    
    df['rdistance'] = (df['to_send'] + ' ' + df['generated']).progress_apply(get_logits_rdistance, model=model, tokenizer=tokenizer, device=device)
    for csize in df["cycle_size"].unique():
        graph_df = df[df["cycle_size"] == csize]
        y=[]
        error_bars=[]
        max_entropy_len = max(graph_df["rdistance"].apply(len))
        for i in range(-10, max_entropy_len):
            # find at which index the cycle begins
            graph_df["generated_from_cycle_start"] = graph_df.apply(lambda row: row["generated"][row["generated"].find(str(row["cycle"])):] if row["cycle_size"] > 0 else row["generated"], axis=1)
            graph_df["toked_gen_cycle"] = graph_df["generated_from_cycle_start"].apply(lambda x: tokenizer([x], return_tensors="pt")["input_ids"][0].cpu().numpy())
            graph_df["cycle_start_index"] = graph_df.apply(lambda row: len(row["rdistance"]) - len(row["toked_gen_cycle"]), axis=1)
            error = np.std(graph_df.apply(lambda x: x["rdistance"][i + x["cycle_start_index"]] if i + x["cycle_start_index"] < len(x["rdistance"]) else None, axis=1).dropna())
            y_i = np.mean(graph_df.apply(lambda x: x["rdistance"][i + x["cycle_start_index"]] if i + x["cycle_start_index"] < len(x["rdistance"]) else None, axis=1).dropna())
            y.append(y_i)
            error_bars.append(error)
        

        plt.plot(range(-10, max_entropy_len), y, label=f"Cycle size: {csize} (n={len(graph_df)})")
        plt.fill_between(range(-10, max_entropy_len), np.array(y) - np.array(error_bars), np.array(y) + np.array(error_bars), alpha=0.2)

    # plt.axvline(x=0, color='r', linestyle='--', label="cycle start", alpha=0.5)
    plt.legend()
    plt.xlabel("Token position in sequence")
    plt.ylabel("Absolute rank distance")
    plt.savefig("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/rdistance_plot.png")
    # plt.savefig("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/rdistance_plot.png")
    plt.clf()