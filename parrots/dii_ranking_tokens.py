import pandas as pd
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import typer
from dadapy import DiffImbalance

# Initialize tqdm for pandas
tqdm.pandas()
def find_seq_idx(short_seq, long_seq):
    """
    Find the index of the first occurrence of short_seq in long_seq.
    Example: find_seq_idx([1, 2], [0, 1, 2, 3, 4]) -> 1
    # can be replaced by np.where(np.correlate(long_seq, short_seq, mode='valid') == len(short_seq))[0][0] ?
    # or by KMP algorithm
    """
    for i in range(len(long_seq) - len(short_seq) + 1):
        if (long_seq[i:i + len(short_seq)] == short_seq).all():
            return i
    return -1

def dii(data_A, data_B, num_points_rows=50, num_epochs=50):
    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A=data_A, # matrix of shape (N,D_A)
        data_B=data_B, # matrix of shape (N,D_B)
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=num_epochs, # number of training epochs
        batches_per_epoch=1, # no mini-batches
        l1_strength=0.0, # no l1 regularization
        point_adapt_lambda=True,
        k_init=1,
        k_final=1,
        # params_init=None, # automatically set to [0.1,0.1,0.1,0.1,0.1]
        optimizer_name="adam", # possible choices: "sgd", "adam", "adamw"
        learning_rate=1e-2,
        learning_rate_decay=None, # possible choices: None, "cos", "exp"
        num_points_rows=num_points_rows, # sample instead of using all points
    )
    _, imbs = dii.train()
    return imbs[-1]

def extract_prob(toked_text, model, device):
    """extract last layer hidden states from a model"""
    _t = {"input_ids": torch.tensor([toked_text]).to(device)}
    o = model(**_t)
    all_logs = o.logits[0, :, :].cpu().detach().numpy()
    # Apply softmax to get probabilities
    probs = np.exp(all_logs) / np.sum(np.exp(all_logs), axis=-1, keepdims=True)
    return probs


if __name__ == "__main__":
    # PARAMS
    model_name = "EleutherAI/pythia-1.4b"
    base_path = "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations"
    # DATA (packing, to be removed?)
    cycle_size = 5
    df = pd.DataFrame()
    for f in Path(base_path).glob(f"cycle_{cycle_size}_results_*.csv"):
        df = pd.concat([df, pd.read_csv(f)])
    df = df[df["cycle"] != "  "]
    df = df[df["cycle_size"] != 0].sample(20)
    df["toked_input"] = df["toked_input"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["cycle"] = df["cycle"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    print("Number of samples:", len(df))

    # INITIALIZE MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df["toked_generated"] = df["generated"].progress_apply(lambda x: tokenizer(x, return_tensors="pt")["input_ids"][0].cpu().numpy())

    df["toked_to_send"] = df.apply(lambda row: np.concatenate([row["toked_input"], row["toked_generated"]]), axis=1)
    df['probs'] = df["toked_to_send"].progress_apply(extract_prob, model=model, device=device)

    # find at which index the cycle begins
    df["cycle_start_index"] = df.progress_apply(lambda row: find_seq_idx(row["cycle"], row["toked_to_send"]), axis=1)
    y=[]
    error_bars=[]
    max_n_cycles = 5
    # max_seq_len * max_seq_len correlation matrix
    correlation_matrix = np.zeros((max_n_cycles, max_n_cycles))
    hub_matrix = np.zeros((max_n_cycles, max_n_cycles))
    for j in range(max_n_cycles):
        for k in range(max_n_cycles):
            probs_j = df.apply(lambda x: x["probs"][x["cycle_start_index"] + j*x["cycle_size"]] if x["cycle_start_index"] + j*x["cycle_size"] < len(x["probs"]) else None, axis=1).tolist()
            probs_k = df.apply(lambda x: x["probs"][x["cycle_start_index"] + k*x["cycle_size"]] if x["cycle_start_index"] + k*x["cycle_size"] < len(x["probs"]) else None, axis=1).tolist()
            # get rid of None values
            idx = [i for i in range(len(probs_j)) if probs_j[i] is not None and probs_k[i] is not None]
            probs_j = [probs_j[i] for i in idx]
            probs_k = [probs_k[i] for i in idx]
            probs_j = np.stack(probs_j)
            probs_k = np.stack(probs_k)

            
            correlation_matrix[j, k] = dii(probs_j, probs_k, num_points_rows=20, num_epochs=50)
            # hub_matrix[j, k] = len(set(map(tuple, toks_j)).intersection(set(map(tuple, toks_k)))) / len(toks_j) if len(toks_j) > 0 else 0
    plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
    # plt.savefig(f"/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/correlation_matrix_csize_{csize}_i_{i}.png")
    plt.savefig(Path(base_path) / f"dii_matrix.png")
    plt.close()
    plt.clf()
    # plt.imshow(hub_matrix, cmap='viridis', interpolation='nearest')
    # plt.savefig(f"/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/hub_matrix_csize_{csize}_i_{i}.png")
    # plt.savefig(Path(base_path) / f"hub_matrix_csize_{csize}_i_{i}.png")
    # plt.close()
    # plt.clf()

        
