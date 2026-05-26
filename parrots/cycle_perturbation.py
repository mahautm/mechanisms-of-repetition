from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/"
    df = pd.read_csv(Path(base_path) / "slot_filling_results_with_cycles.csv")

    # Load the model and tokenizer
    model_name = "EleutherAI/pythia-1.4B" # "mistralai/Mistral-7B-v0.3"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # read file /projects/colt/tagged_pile_10k/tagged_pile_10k.txt
    with open("/projects/colt/tagged_pile_10k/tagged_pile_10k.txt") as f:
        lines = f.readlines()

    # get the 1000 Nouns, tagged NOUN
    nouns = [line.split("\t")[0] for line in lines if "\t" in line and line.split("\t")[1].strip() == "NOUN"]
    # select 1000 random nouns
    nouns = np.random.choice(nouns, 1000)
    # get the 1000 Verbs, tagged VERB
    verbs = [line.split("\t")[0] for line in lines if "\t" in line and line.split("\t")[1].strip() == "VERB"]
    # select 1000 random verbs
    verbs = np.random.choice(verbs, 1000)
    # get the 1000 Adjectives, tagged ADJ
    adjs = [line.split("\t")[0] for line in lines if "\t" in line and line.split("\t")[1].strip() == "ADJ"]
    # select 1000 random adjectives
    adjs = np.random.choice(adjs, 1000)

    df = df[df["cycle"] != "  "]
    df = df[df["cycle_size"] >= 2]
    df["transition"] = df.apply(lambda row: row["generated"][:row["generated"].find(str(row["cycle"]))] if row["cycle_size"] > 0 else row["generated"], axis=1)
    inputs = (df["to_send"] + df["transition"]).tolist()
    print(inputs[1:4])
    print(df["generated"][1:4].tolist())
    print(df["to_send"][1:4].tolist())
    cycles = df["cycle"].tolist()
    print(cycles[1:4])

    max_cycles = 100
    batch_size = 100
    for cycle_idx in range(max_cycles):
        for batch_idx in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch_idx:batch_idx+batch_size]
            batch_cycles = cycles[batch_idx:batch_idx+batch_size]
            prepared_inputs = [i + c * cycle_idx for i, c in zip(batch_inputs, batch_cycles)]
            toked = tokenizer(prepared_inputs, return_tensors="pt", padding=True).to(device)
            outputs = model.generate(**toked, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            # is the cycle in the NEW generated text?
            detok = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            cycle_in_output = [str(c) in d.split(p)[1] for c, d, p in zip(batch_cycles, detok, prepared_inputs)]
            print(sum(cycle_in_output)/len(cycle_in_output))

if __name__ == "__main__":
    main()
