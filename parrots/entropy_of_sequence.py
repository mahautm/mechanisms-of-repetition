import pandas as pd
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
# Initialize tqdm for pandas
tqdm.pandas()

# This is used multiple times and maybe should go into utils
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

def calculate_entropy(logits):
    """Calculate the entropy of the logits."""
    logits = logits - np.max(logits, axis=1, keepdims=True)  # for numerical stability
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return entropy(probabilities, axis=1)

def get_logits_batch(tokens, model, tokenizer, device, batch_size=8):
    """Extract logits from the model for each sequence in the DataFrame using batching."""
    all_logits = []
    for i in tqdm(range(0, len(tokens), batch_size), desc="Processing batches"):
        batch_tokens = tokens[i:i + batch_size]
        # pad the batch using numpy and tokenizer.pad_token_id
        max_len = max(len(x) for x in batch_tokens)
        batch_tokens = np.array([np.pad(x, (0, max_len - len(x)), constant_values=tokenizer.pad_token_id) for x in batch_tokens])
        batch_tokens = torch.tensor(batch_tokens).to(device).to(torch.int64)
        with torch.no_grad():
            outputs = model(batch_tokens)
        logits = outputs.logits.cpu().numpy()
        for j in range(len(batch_tokens)):
            padding_token=tokenizer.pad_token_id
            # Find the position of the first padding token
            padding_position = np.where(batch_tokens[j].cpu() == padding_token)[0]
            # If there are padding tokens, remove the logits after the first padding token
            if len(padding_position) > 0:
                all_logits.append(logits[j, :padding_position[0]])
            else:
                all_logits.append(logits[j])
    return all_logits

def get_entropy_data(df, model, tokenizer, device, icl:bool=False):
    df = df.copy()
    # Create a sample DataFrame
    n_cycles = 50
    if icl:
        df["toked_to_send"] = df["cycle"].apply(lambda x: np.tile(x, n_cycles)[:300])
        df["cycle_start_index"] = 0

    else:
        df["toked_to_send"] = df["generated"].progress_apply(lambda x: tokenizer(x, return_tensors="pt")["input_ids"][0].cpu().numpy())
        # df["toked_to_send"] = df.apply(lambda row: np.concatenate([row["toked_input"], row["toked_generated"]]), axis=1)
        df["cycle_start_index"] = df.progress_apply(lambda row: find_seq_idx(row["cycle"], row["toked_to_send"]), axis=1)
        df= df[df["cycle_start_index"] != -1]
        df= df[df["cycle_start_index"] != 0]

    df['logits'] = get_logits_batch(df['toked_to_send'].tolist(), model=model, tokenizer=tokenizer, device=device, batch_size=8)
    df['entropy'] = df["logits"].progress_apply(calculate_entropy)
    # ensure cycle_start_index and cycle_size are int, not float
    df["cycle_start_index"] = df["cycle_start_index"].astype(int)
    print("Number of NaN values:", df["cycle_size"].isna().sum())
    
    df["cycle_size"] = df["cycle_size"].fillna(-1).astype(int)    # average the entropy over the cycles
    df["entropy"] = df.apply(lambda x: [
            np.array(x["entropy"][x["cycle_start_index"] + i: x["cycle_start_index"]+i+x["cycle_size"]]).mean() for i in range(x["cycle_start_index"], len(x["entropy"]), x["cycle_size"]) if np.any(x["entropy"][x["cycle_start_index"] + i: x["cycle_start_index"]+i+x["cycle_size"]])
        ], axis=1)
    # cutoff after 50 cycles
    df["entropy"] = df["entropy"].apply(lambda x: x[:10])

    max_entropy_len = df["entropy"].apply(len).max().astype(int)
    df["entropy"] = df["entropy"].apply(lambda x: np.pad(x, (0, max_entropy_len - len(x)), constant_values=np.nan))
    return df, max_entropy_len
# Example usage
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    df, max_entropy_len = get_entropy_data(model, tokenizer, icl=True)
    # Graph
    y = []
    y_std = []
    graph_df = pd.DataFrame(columns=["x", "y"])
    for i in range(max_entropy_len):
        _temp_df = pd.DataFrame(columns=["x", "y"])
        _temp_df["y"] = df["entropy"].apply(lambda x: x[i] if i < len(x) else None).dropna()
        _temp_df["x"] = i
        graph_df = pd.concat([graph_df, _temp_df])
    sns.lineplot(data=graph_df, x="x", y="y")
    plt.ylabel("Entropy")
    plt.xlabel("Cycle nº")
    plt.legend()
    plt.savefig("/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations/entropy_icl.png")
