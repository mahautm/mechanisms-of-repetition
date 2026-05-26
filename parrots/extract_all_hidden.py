# from Marco modified by me
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pickle
import sys
import typer
from typing import List, Optional
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import numpy as np

def model_pass(raw_inputs, tokenizer, model, device, save_attention=False):
    # for now, I don't constraint to a max length
    inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt", padding_side="left").to(device)

    with torch.no_grad():
        g = model(**inputs,output_hidden_states=True,output_attentions=True)
        hidden_states = g.hidden_states
        attentions = g.attentions

    
    per_layer_activations = []
    for layer_idx, raw_activation in enumerate(hidden_states): 
        # shape of hidden_states: layers x batch_size x tokens x d
        last_token_activations = []
        for i in range(len(raw_inputs)):
            # traversing batch items
            print(f"raw_activation.shape: {raw_activation.shape}")
            last_token_activation = raw_activation[i, -1].cpu().numpy()
            last_token_activations.append(last_token_activation)
        per_layer_activations.append(last_token_activations)

    if save_attention:
        # shape of attention: layers * batch_size * num_heads * sequence_length * sequence_length
        per_layer_attentions = []
        for layer_idx, raw_attention in enumerate(attentions):
            last_attentions = []
            for i in range(len(raw_inputs)):
                print(f"raw_attention[layer_idx].shape: {raw_attention[layer_idx].shape}")
                last_attention = raw_attention[layer_idx,i].cpu().numpy()
                last_attentions.append(last_attention)
            per_layer_attentions.append(last_attentions)
    else:
        per_layer_attentions = None

    return(per_layer_activations, per_layer_attentions)

def main(
    model_name:str,
    batch_size:int,
    data_file:str, 
    save_dir:str="",
    save_size:int=1024,
    n_examples:int=10000, # number of examples to keep from the dataset, if None, keep all
    n_cycles:int=4,
    save_attention:bool=False,
    seed:int=42,
    ):

    # INITIALIZATION
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DATA LOADING
    if ".csv" in data_file:
        inputs = pd.read_csv(data_file)

        # select only cycles
        inputs = inputs[inputs["cycle_size"]!=0]
        inputs = inputs[inputs["cycle_count"]>=n_cycles]
        cycle_size = inputs["cycle_size"].tolist()
        cycles = inputs["cycle"].tolist()

        inputs["generated_before_cycle"] = inputs.apply(lambda row: row["to_send"] + row["generated"][:row["generated"].find(str(row["cycle"]))], axis=1)
        inputs=inputs["generated_before_cycle"]

        # only keep n_examples
        if n_examples is not None and n_examples < len(inputs):
            # DANGER sampling after all the lists were made, nothing will work
            assert False, "This is not implemented yet"
            inputs = inputs.sample(n=n_examples, random_state=seed).tolist()
        else:
            inputs = inputs.tolist()
        print(f"Number of datapoints : {len(inputs)}")
        # in the case of ICL, we just need to start with a cycle
        # inputs = cycles

    else:
        raise ValueError("data_file should be a csv")

    # SAVE processed inputs for verification
    with open(Path(save_dir) / "inputs.txt", "w") as f:
        f.write("\n".join(inputs))

    # EXTRACTING HIDDEN STATES

    states = dict()
    attentions = dict()
    
    for cycle_idx in range(n_cycles + 1):
        # deal with batching by hand - including the final batch which might be smaller
        start_index = 0
        current_batch_size = batch_size
        if cycle_idx!=0:
            inputs = [inputs[i] + cycles[i] for i in range(len(inputs))]

        while ((start_index+current_batch_size)<len(inputs)):
            if (current_batch_size + start_index) > len(inputs):
                current_batch_size = len(inputs)

            layer_output, attention_output = model_pass(inputs[start_index:start_index+current_batch_size], tokenizer, model, device, save_attention=save_attention)

            for i in range(len(layer_output)):
                if not i in states:
                    states[i] = []
                states[i] = states[i] + layer_output[i]

            if save_attention:
                for i in range(len(attention_output)):
                    if not i in attentions:
                        attentions[i] = []
                    attentions[i] = attentions[i] + attention_output[i]

            start_index=start_index+current_batch_size

            # SAVE STATES every save_size samples
            if (start_index // save_size) > ((start_index - current_batch_size) // save_size) or (start_index + current_batch_size) >= len(inputs):
                for layer_idx in states:
                    layer_save_dir = Path(save_dir) / f"layer_{layer_idx}"
                    layer_save_dir.mkdir(parents=True, exist_ok=True)
                    with open(layer_save_dir / f"hidden_states_cycle_{cycle_idx}_{start_index // save_size}.pkl", "wb") as f:
                        pickle.dump(states[layer_idx], f)
                if save_attention:
                    for layer_idx in attentions:
                        layer_save_dir = Path(save_dir) / f"layer_{layer_idx}"
                        layer_save_dir.mkdir(parents=True, exist_ok=True)
                        with open(layer_save_dir / f"attentions_cycle_{cycle_idx}_{start_index // save_size}.pkl", "wb") as f:
                            pickle.dump(attentions[layer_idx], f)
                states = dict()
                attentions = dict()

if __name__ == "__main__":
    typer.run(main)
