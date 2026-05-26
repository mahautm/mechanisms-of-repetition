import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from parrots.cycle_detection import detect_cycles
from tqdm import tqdm
from rouge_score import rouge_scorer
import typer

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

def initialize(model_name:str, load_in_4bit:bool=False):
    base_path:str=f"/home/mmahaut/projects/parrots/outputs/{model_name}_human_lama_parrots_list_v1_sf/perturbations"

    df = pd.DataFrame()
    for f in Path(base_path).glob(f"cycle_{3}_results_*.csv"):
        df = pd.concat([df, pd.read_csv(f)], ignore_index=True)

    df["toked_input"] = df["toked_input"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["cycle"] = df["cycle"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["toked_transition"] = df["toked_transition"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df = df[df["cycle_size"] != 0]
    df = df[df["cycle_size"] != 1]
    df = df[df["cycle_count"] != 0]
    df = df[df["cycle_count"] != 1]
    # df = df[df["cycle_size"] < 10]
    # df=df.sample(1000)
    print(f"Loaded {len(df)} rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=load_in_4bit).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return df, model, tokenizer, device

def complete_cycle(generation, cycle):
    """
    Check if once the cycle appears in the generation, there are nothing but cycles until the end of the generation
    ex: complete_cycle([0,0,0,1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 2, 3]) -> True
    complete_cycle([0,0,0,1, 2, 3, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3]) -> False
    
    the final cycle does not need to be complete
    ex: complete_cycle([0,0,0,1, 2, 3, 1, 2, 3, 1, 2,], [1, 2, 3]) -> True
    """
    cycle = np.array(cycle)
    generation = np.array(generation)
    for i in range(len(generation)):
        if (generation[i:i + len(cycle)] == cycle[:len(generation[i:i + len(cycle)])]).all():
            for j in range(i + len(cycle), len(generation), len(cycle)):
                if not (generation[j:j + len(cycle)] == cycle[:len(generation[j:j + len(cycle)])]).all():
                    return False
            return True
    return False

def get_topp_data(topp:float, icl:bool=False, n_cycles:int=1, model_name:str="EleutherAI/pythia-1.4B", load_in_4bit:bool=True):
    df, model, tokenizer, device = initialize(model_name, load_in_4bit)
    df = df.copy()

    if icl:
        df["toked_to_send"] = df["cycle"].apply(lambda x: np.tile(x, n_cycles + 1))
    else:
        df["toked_to_send"] = df.apply(lambda row: np.concatenate([row["toked_input"], row["toked_transition"], np.tile(row["cycle"], n_cycles)]), axis=1)
    df[f"top-p-{topp}-gen"] = generate_batch(df, topp, tokenizer, model, device)
    _temp = df[f"top-p-{topp}-gen"].apply(lambda x: detect_cycles(x) if x is not None else (None, None, None))
    df[f"top-p-{topp}-gen-cycle"], df[f"top-p-{topp}-gen-cycle-size"], df[f"top-p-{topp}-gen-cycle-count"] = zip(*_temp)
    # proportion of sentence which is made of repetitions
    y_prop_in_generations = df.apply(lambda row: row[f"top-p-{topp}-gen-cycle-count"] * row[f"top-p-{topp}-gen-cycle-size"] / len(row[f"top-p-{topp}-gen"]), axis=1).mean()
    
    # proportion of cycles detected in top-p sampling
    y_cycles = df[f"top-p-{topp}-gen-cycle-size"].apply(lambda x: x > 0).sum() / len(df)
    y_complete_cycles = df.apply(lambda row: complete_cycle(row[f"top-p-{topp}-gen"], row["cycle"]), axis=1).sum() / len(df)
    # proportion of cycles that are the same as in the perturbation
    df = df.drop(columns=[f"top-p-{topp}-gen"])

    def shared_tokens(row, topp):
        if row[f"top-p-{topp}-gen-cycle-size"] == 0:
            return 0
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(' '.join(map(str, row["cycle"])), ' '.join(map(str, row[f"top-p-{topp}-gen-cycle"])))
        return score['rougeL'].fmeasure
    
    y_shared_tokens = df.apply(lambda row: shared_tokens(row, topp), axis=1).mean()
    print(f"Top-p {topp}")
    print(f"n_cycles: {n_cycles}")
    print(f"ICL: {icl}")
    print(f"Proportion of cycles detected in top-p sampling: {y_cycles}")
    print(f"Proportion of complete cycles detected in top-p sampling: {y_complete_cycles}")
    print(f"ROUGE-L between greedy and top-p cycles: {y_shared_tokens}")
    print(f"Proportion of sentence which is made of repetitions: {y_prop_in_generations}")
    return y_cycles, y_shared_tokens, y_prop_in_generations

if __name__ == "__main__":
    typer.run(get_topp_data)


    