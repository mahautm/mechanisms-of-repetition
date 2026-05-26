import pandas as pd
import numpy as np
from parrots.entropy_of_sequence import get_entropy_data
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

def get_logits_batch(tokens, model, tokenizer, device, batch_size=8):
    """Extract logits from the model for each sequence in the DataFrame using batching."""
    all_logits = []
    for i in tqdm(range(0, len(tokens), batch_size), desc="Processing batches"):
        batch_tokens = tokens[i:i + batch_size]
        # pad the batch using numpy and tokenizer.pad_token_id
        max_len = max(len(x) for x in batch_tokens)
        batch_tokens = np.array([np.pad(x, (0, max_len - len(x)), constant_values=tokenizer.pad_token_id) for x in batch_tokens])
        batch_tokens = torch.tensor(batch_tokens).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens)
        logits = outputs.logits.cpu().numpy()
        # softmax
        logits = np.exp(logits - np.max(logits, axis=2, keepdims=True)) / np.sum(np.exp(logits - np.max(logits, axis=2, keepdims=True)), axis=2, keepdims=True)
        # only keep probabilities of the selected tokens
        logits = np.take_along_axis(logits, batch_tokens[:, :, None].cpu().numpy(), axis=2).squeeze()
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

def get_prob_data(df, model, tokenizer, device, icl:bool=False, n_cycles:int=50):
    df = df.copy()
    if icl:
        df["toked_to_send"] = df["cycle"].apply(lambda x: np.tile(x, n_cycles)[:500])
        df["cycle_start_index"] = 0
    else:
        df["toked_to_send"] = df["generated"].progress_apply(lambda x: tokenizer(x, return_tensors="pt")["input_ids"][0].cpu().numpy())
        # df["toked_to_send"] = df.apply(lambda row: np.concatenate([row["toked_input"], row["toked_generated"]]), axis=1)
        df["cycle_start_index"] = df.progress_apply(lambda row: find_seq_idx(row["cycle"], row["toked_to_send"]), axis=1)
        df= df[df["cycle_start_index"] != -1]
        df= df[df["cycle_start_index"] != 0]

    df['logits'] = get_logits_batch(df['toked_to_send'].tolist(), model=model, tokenizer=tokenizer, device=device, batch_size=8)
    df["logits"] = df["logits"].apply(lambda x: x if (isinstance(x, list) or isinstance(x, np.ndarray)) else [None])
    # ensure cycle_start_index and cycle_size are int, not float
    df["cycle_start_index"] = df["cycle_start_index"].astype(int)
    df["cycle_size"] = df["cycle_size"].astype(int)
    # average the logits over the cycles
    df["logits"] = df.apply(lambda x: [
            np.array(x["logits"][x["cycle_start_index"] + i: x["cycle_start_index"]+i+x["cycle_size"]]).mean() for i in range(x["cycle_start_index"], len(x["logits"]), x["cycle_size"]) if np.any(x["logits"][x["cycle_start_index"] + i: x["cycle_start_index"]+i+x["cycle_size"]])
        ], axis=1)
    df["logits"] = df["logits"].apply(lambda x: x[:10])

    max_entropy_len = df["logits"].apply(len).max().astype(int)
    df["logits"] = df["logits"].apply(lambda x: np.pad(x, (0, max_entropy_len - len(x)), constant_values=np.nan))
    # print(df["logits"].head())
    return df, max_entropy_len
# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "Qwen/Qwen2.5-7B"
    model_name = "EleutherAI/pythia-1.4b"
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a sample DataFrame
    base_path:str=f"/home/mmahaut/projects/parrots/outputs/{model_name}_human_lama_parrots_list_v1_sf/perturbations"
    df = pd.DataFrame()
    for f in Path(base_path).glob(f"cycle_{3}_results_*.csv"):
        df = pd.concat([df, pd.read_csv(f)])
    df = df[df["cycle_size"] != 0]
    df = df[df["cycle_size"] != 1]
    df = df[df["cycle_count"] != 0]
    df = df[df["cycle_count"] != 1]
    df = df[df["cycle_size"] < 10]
    # df = df.sample(100)
    print("Number of samples:", len(df))

    df["toked_input"] = df["toked_input"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["cycle"] = df["cycle"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])

    # Graph
    df1, max_entropy_len1 = get_prob_data(df, model, tokenizer, device, icl=True)
    df1 = df1[["logits", "cycle_size"]]
    df2, max_entropy_len2 = get_prob_data(df, model, tokenizer, device, icl=False)
    df2 = df2[["logits", "cycle_size"]]

    dfe1, _ = get_entropy_data(df, model, tokenizer, device, icl=True)
    dfe1 = dfe1[["entropy", "cycle_size"]]
    dfe2, _ = get_entropy_data(df, model, tokenizer, device, icl=False)
    dfe2 = dfe2[["entropy", "cycle_size"]]

    graph_df = pd.DataFrame(columns=["x", "y", "entropy", "Legend", "cycle_size"])
    for i in range(max_entropy_len1):
        _temp_df = pd.DataFrame(columns=["x", "y", "entropy", "Legend", "cycle_size"])
        # check if the index is duplicate
        _temp_df["y"] = df1["logits"].apply(lambda x: x[i] if i < len(x) else None)
        _temp_df["entropy"] = dfe1["entropy"].apply(lambda x: x[i] if i < len(x) else None)
        _temp_df["Legend"] = "ICL"
        _temp_df["x"] = i
        _temp_df["cycle_size"] = df1["cycle_size"]
        graph_df = pd.concat([graph_df, _temp_df], ignore_index=True)
    for i in range(max_entropy_len2):
        _temp_df = pd.DataFrame(columns=["x", "y", "entropy", "Legend", "cycle_size"])
        _temp_df["y"] = df2["logits"].apply(lambda x: x[i] if i < len(x) else None)
        _temp_df["entropy"] = dfe2["entropy"].apply(lambda x: x[i] if i < len(x) else None)
        _temp_df["Legend"] = "Natural"
        _temp_df["x"] = i
        _temp_df["cycle_size"] = df2["cycle_size"]
        graph_df = pd.concat([graph_df, _temp_df], ignore_index=True)

    print(graph_df.head())
    graph_df.rename(columns={"x": "Cycle", "y": "Probability", "entropy": "Entropy", "cycle_size": "Cycle Size"}, inplace=True)
    # markers1 = {"ICL": "o", "Natural": "s"}
    # markers2 = {"ICL": "*", "Natural": "x"}

    # palette_dict1 = {"ICL": "blue", "Natural": "red"}
    # palette_dict2 = {"ICL": "orange", "Natural": "green"}
    # sns.lineplot(data=graph_df, x="X", y="Probability", hue="Legend", markers=markers1, palette=palette_dict1)
    # plt.ylabel("Probability")

    # put the scale of the second plot on the right side
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(data=graph_df, x="Cycle", y="Probability", hue="Legend", style="Legend", markers=True, dashes=False, ax=ax1, legend=False)

    ax1.set_ylabel("Probability", fontsize=14)
    ax1.set_xlabel("Cycle", fontsize=14)

    # ax1.legend(fontsize=14)

    sns.lineplot(data=graph_df, x="Cycle", y="Entropy", hue="Legend",style="Legend", ax=ax2, markers=True, dashes=False)
    ax2.legend(fontsize=14)
    # sns.lineplot(data=graph_df, x="X", y="Entropy", hue="Legend", markers=markers2, ax=ax2, palette=palette_dict2)
    # ax2.set_ylabel("Entropy")
    ax2.set_ylabel("Entropy", fontsize=14)
    ax2.set_xlabel("Cycle", fontsize=14)


    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/home/mmahaut/projects/parrots/outputs/{model_name}_human_lama_parrots_list_v1_sf/perturbations/probability_all.png")
