import pandas as pd
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

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

def top_k_tokens(toked_text, n_tokens, model, tokenizer, device):
    """Extract top k most probable generations for every token in a sequence."""
    toked = {"input_ids": torch.tensor([toked_text], dtype=torch.long).to(device), "attention_mask": torch.ones((1, len(toked_text)), dtype=torch.long).to(device)}
    o = model(**toked)
    all_logs = o.logits[0, :, :].cpu().detach().numpy()
    top_k_tokens = []
    for i in range(1, all_logs.shape[0]):
        top_k = all_logs[i, :].argsort()[-n_tokens:]
        top_k_tokens.append(tokenizer.convert_ids_to_tokens(top_k))
    return top_k_tokens

# Example usage

# Example usage
if __name__ == "__main__":
    model_name = "EleutherAI/pythia-1.4b"
    # Initialize the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # base_path = "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/"
    # df=pd.read_csv(Path(base_path) / "slot_filling_results_with_cycles.csv")
    # # df = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results_with_cycles.csv")
    # # df=pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results_with_cycles.csv")
    # df = df[df["cycle"] != "  "]
    # df = df[df["cycle_size"] <= 6]
    # # sample 50 rows from each cycle size
    # df = df.groupby("cycle_size").apply(lambda x: x.sample(min(100, len(x)))).reset_index(drop=True)

    # Pile Data
    base_path:str="/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations"
    cycle_size = 5
    df = pd.DataFrame()
    for f in Path(base_path).glob(f"cycle_{cycle_size}_results_*.csv"):
        df = pd.concat([df, pd.read_csv(f)])
    df = df[df["cycle_size"] != 0]
    df = df[df["cycle_size"] != 1]

    df["toked_input"] = df["toked_input"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["cycle"] = df["cycle"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])

    df["toked_generated"] = df["generated"].progress_apply(lambda x: tokenizer(x, return_tensors="pt")["input_ids"][0].cpu().numpy())
    df["toked_to_send"] = df.apply(lambda row: np.concatenate([row["toked_input"], row["toked_generated"]]), axis=1)
    # find at which index the cycle begins
    df["cycle_start_index"] = df.progress_apply(lambda row: find_seq_idx(row["cycle"], row["toked_to_send"]), axis=1)
    df= df[df["cycle_start_index"] != -1]
    df= df[df["cycle_start_index"] != 0]

    print("Number of samples:", len(df))
    

    df['top_5_tokens'] = (df["toked_to_send"]).progress_apply(top_k_tokens, n_tokens=5, model=model, tokenizer=tokenizer, device=device)
    for csize in tqdm(df["cycle_size"].unique(), desc="Cycle size"):
        graph_df = df[df["cycle_size"] == csize]
        # find at which index the cycle begins
        # graph_df["generated_from_cycle_start"] = graph_df.apply(lambda row: row["generated"][row["generated"].find(str(row["cycle"])):] if row["cycle_size"] > 0 else row["generated"], axis=1)
        # graph_df["toked_gen_cycle"] = graph_df["generated_from_cycle_start"].apply(lambda x: tokenizer([x], return_tensors="pt")["input_ids"][0].cpu().numpy())
        # graph_df["cycle_start_index"] = graph_df.apply(lambda row: len(row["top_5_tokens"]) - len(row["toked_gen_cycle"]), axis=1)
        y1=[]
        y2=[]
        max_seq_len = max(graph_df["top_5_tokens"].apply(len))
        # max_seq_len * max_seq_len correlation matrix
        correlation_matrix = np.zeros((max_seq_len, max_seq_len))
        hub_matrix = np.zeros((max_seq_len, max_seq_len))
        # for i in range(csize):
            # correlation_matrix = np.zeros((max_seq_len, max_seq_len))
            # hub_matrix = np.zeros((max_seq_len, max_seq_len))
        for j in range(0, max_seq_len):
            for k in range(0, max_seq_len):
                toks_j = graph_df.apply(lambda x: x["top_5_tokens"][x["cycle_start_index"] + j] if x["cycle_start_index"] + j < len(x["top_5_tokens"]) else None, axis=1).dropna()
                toks_k = graph_df.apply(lambda x: x["top_5_tokens"][x["cycle_start_index"] + k] if x["cycle_start_index"] + k < len(x["top_5_tokens"]) else None, axis=1).dropna()
                correlation_matrix[j, k] = sum([len(set(toks_j).intersection(set(toks_k))) for toks_j, toks_k in zip(toks_j, toks_k)]) / len(toks_j) if len(toks_j) > 0 else 0
                hub_matrix[j, k] = len(set(map(tuple, toks_j)).intersection(set(map(tuple, toks_k)))) / len(toks_j) if len(toks_j) > 0 else 0
            
            # plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
            # plt.savefig(Path(base_path) / f"correlation_matrix_csize_{csize}_i_{i}.png")
            # plt.close()
            # plt.clf()
        y1.append(correlation_matrix.sum(axis=0)/len([x for x in correlation_matrix if x is not None]))
        y2.append(hub_matrix.sum(axis=0)/len([x for x in hub_matrix if x is not None]))

        plt.plot(correlation_matrix.sum(axis=0), label="correlation")
        plt.plot(hub_matrix.sum(axis=0), label="hub")
        plt.legend()
        plt.savefig(Path(base_path) / f"correlation_hub_csize_{csize}.png")
        plt.close()
            
            # plt.imshow(hub_matrix, cmap='viridis', interpolation='nearest')
            # plt.savefig(Path(base_path) / f"hub_matrix_csize_{csize}_i_{i}.png")
            # plt.close()
            # plt.clf()

            
