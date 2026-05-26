import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from parrots.cycle_detection import detect_cycles
from tqdm import tqdm

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

def generate_batch(df, topp, tokenizer, model, device, batch_size=32, max_new_tokens=350):
    generated = []
    for i in tqdm(range(0, len(df), batch_size), desc=f"Top-p {topp}"):
        max_length = max(len(seq) for seq in df["toked_to_send"].iloc[i:i + batch_size])
        batch = df["toked_to_send"].iloc[i:i + batch_size].tolist()
        batch = np.array([np.pad(seq, (max_length - len(seq), 0), constant_values=tokenizer.pad_token_id) for seq in batch])
        inputs = torch.tensor(batch, dtype=torch.long).to(device)
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)
        gen = model.generate(input_ids=inputs, attention_mask=attention_mask, top_p=topp, do_sample=True, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        for g in gen:
            generated.append(g[len(inputs[0]):].cpu().numpy())
    return generated

if __name__ == "__main__":
    base_path:str="/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations"
    n_cycles = 2
    icl = True
    df = pd.DataFrame()
    for f in Path(base_path).glob(f"cycle_{3}_results_*.csv"):
        df = pd.concat([df, pd.read_csv(f)])

    df["toked_input"] = df["toked_input"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["cycle"] = df["cycle"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["toked_transition"] = df["toked_transition"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df = df[df["cycle_size"] != 0]
    df = df[df["cycle_size"] != 1]
    df = df[df["cycle_count"] != 0]
    df = df[df["cycle_count"] != 1]
    df = df[df["cycle_size"] < 10]
    df=df.sample(100)

    # table has columns named entropy,perturbator_prob,rank,perturbator_token,prev_cycle,cycle,cycle_size,cycle_count,generation
    # we want to plot log of perturbator prob as x and entropy as y
    # colour depends on if prev_cycle is equal to cycle

    # Graph 1 - nothing cool
    # df["equal_cycles"] = df.apply(lambda row: bool(set(row["prev_cycle"]).intersection(set(row["cycle"]))), axis=1)
    # sns.lineplot(data=df, x="perturbator_prob", y="entropy", hue="equal_cycles")
    # plt.ylabel("Entropy")
    # plt.xlabel("Log of perturbator probability")
    # plt.legend()
    # plt.show()
    # plt.savefig("perturbation_graph.png")
    # plt.close()

    # graph 2 - load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # now we generate with top-p sampling and see if there are still cycles + if they're the same
    x=np.arange(0, 1, 0.05)
    for topp in tqdm(x):
        if icl:
            df["toked_to_send"] = df["cycle"].apply(lambda x: np.repeat(x, n_cycles))
        else:
            df["toked_to_send"] = df.apply(lambda row: np.concatenate([row["toked_input"], row["toked_transition"]]), axis=1)
        df[f"top-p-{topp}-gen"] = generate_batch(df, topp, tokenizer, model, device)
        df[f"top-p-{topp}-gen"] = df[f"top-p-{topp}-gen"].apply(lambda x: detect_cycles(x) if x is not None else (None, None, None))
        df[f"top-p-{topp}-gen-cycle"], df[f"top-p-{topp}-gen-cycle-size"], df[f"top-p-{topp}-gen-cycle-count"] = zip(*df[f"top-p-{topp}-gen"])
        df = df.drop(columns=[f"top-p-{topp}-gen"])
    
    # proportion of cycles detected in top-p sampling
    y_cycles = [df[f"top-p-{topp}-gen-cycle-size"].apply(lambda x: x > 0).sum() / len(df) for topp in x]
    # proportion of cycles that are the same as in the perturbation
    def shared_tokens(row, topp):
        if row[f"top-p-{topp}-gen-cycle-size"] == 0:
            return 0
        return len(set(row["cycle"]).intersection(set(row[f"top-p-{topp}-gen-cycle"]))) / len(set(row["cycle"]))
    y_shared_tokens = [df.apply(lambda row: shared_tokens(row, topp), axis=1).mean() for topp in x]
    sns.lineplot(x=x, y=y_cycles, label="Proportion of cycles detected in top-p sampling")
    sns.lineplot(x=x, y=y_shared_tokens, label="Proportion of shared tokens in cycles")
    plt.ylabel("Proportion")
    plt.xlabel("Top-p")
    plt.legend()
    plt.show()
    plt.savefig(f"perturbation_graph_{'icl' if icl else 'natural'}.png")


    