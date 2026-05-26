# from Marco
# BEWARE! only generated is used, not to_send, this was corrected in parrots.extract_all_hidden
# padding only dealt with in one direction, not both - we expect padding to be on the left and then ignore all padding tokens when extracting hidden states.
# in parrots.extract_all_hidden, the padding is on the left and we therefore can take the last token of the sequence.
# variables need to be renamed

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
import logging
import numpy as np

def model_pass(raw_inputs, tokenizer, model, device, save_attention=False):
    # for now, I don't constraint to a max length
    inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt").to(device)

    # making sure we collect hidden state after the last "true" token
    # and not a padding token
    last_true_token_indices = []
    for att_mask in inputs.attention_mask:
        if not(0 in att_mask):
            last_true_token_indices.append(len(att_mask)-1)
        else:
            last_true_token_indices.append(att_mask.tolist().index(0)-1)

    with torch.no_grad():
        g = model(**inputs,output_hidden_states=True,output_attentions=True)
        hidden_states = g.hidden_states
        attentions = g.attentions

    
    per_layer_activations = []
    for layer_idx, raw_activation in enumerate(hidden_states): # shape of hidden_states: layers x batch_size x tokens x d
        # shape of attention: layers * batch_size * num_heads * sequence_length * sequence_length
        # traversing layers
        last_token_activations = []
        for i in range(len(last_true_token_indices)):
            # traversing batch items
            print(f"raw_activation.shape: {raw_activation.shape}")
            last_token_activation = raw_activation[i, last_true_token_indices[i]].cpu().numpy()
            last_token_activations.append(last_token_activation)
        # appending a list of all the last-token-activations of the current layer to a list of lists
        per_layer_activations.append(last_token_activations)
    if save_attention:
        per_layer_attentions = []
        for layer_idx, raw_attention in enumerate(attentions):
            last_attentions = []
            for i in range(len(last_true_token_indices)):
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
    override_data_file:bool=False,
    out_pickle_prefix:str="",
    checkpoint_path:str=None,
    sanity_check:bool=False,
    no_cycles:bool=False,
    kn_threshold:Optional[float]=None,
    save_attention:bool=False
    ):
    Path(out_pickle_prefix).parent.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info("device is " + device, file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if ".csv" in data_file:
        inputs = pd.read_csv(data_file)
        # select cycle type
        if no_cycles:
            inputs = inputs[inputs["cycle_size"]==0]
            inputs= inputs["generated"].tolist()
        else:
            inputs = inputs[inputs["cycle_size"]!=0]
            cycle_size = inputs["cycle_size"].tolist()
            inputs["generated_before_cycle"] = inputs.apply(lambda row: row["generated"][:row["generated"].find(str(row["cycle"]))], axis=1)
            inputs= inputs["generated_before_cycle"].tolist()

    else:
        raise ValueError("data_file should be a csv")
    # save
    with open(out_pickle_prefix + "inputs.txt", "w") as f:
        f.write("\n".join(inputs))
    cases_count = len(inputs)
    first_index = 0
    current_batch_size = batch_size
    states = dict()
    attentions = dict()
    if (current_batch_size>cases_count):
        current_batch_size = cases_count
    while ((first_index+current_batch_size)<cases_count):
        layer_output, attention_output = model_pass(inputs[first_index:first_index+current_batch_size], tokenizer, model, device, save_attention=save_attention)
        for i in range(len(layer_output)):
            if not i in states:
                states[i] = []
            states[i] = states[i] + layer_output[i]
        if save_attention:
            for i in range(len(attention_output)):
                if not i in attentions:
                    attentions[i] = []
                attentions[i] = attentions[i] + attention_output[i]
        first_index=first_index+current_batch_size
    # in case cases_count is not a multiple of batch_size
    # if first_index<cases_count:
    #     layer_output, attention_output = model_pass(inputs[first_index:cases_count], tokenizer, model, device)
    #     for i in range(len(layer_output)):
    #         states[i] = states[i] + layer_output[i]
    #     if save_attention:
    #         for i in range(len(attention_output)):
    #             attentions[i] = attentions[i] + attention_output[i]

    # out_pickle_name = out_pickle_prefix + ".pickle"
    # with open(out_pickle_name, 'wb') as f:
    #     pickle.dump(states, f)
    # if save_attention:
    #     att_pickle_name = out_pickle_prefix + "_att.pickle"
    #     with open(att_pickle_name, 'wb') as f:
    #         pickle.dump(attentions, f)

    # take all the activations from a layer and average them
    
    if no_cycles:
        average_states = dict()
        for layer_idx, activations in states.items():
            average_states[layer_idx] = np.mean(activations, axis=0)
        out_pickle_name = out_pickle_prefix + ".pickle"
        with open(out_pickle_name, 'wb') as f:
            pickle.dump(average_states, f)

        if save_attention:
            average_attentions = dict()
            for layer_idx, _att in attentions.items():
                # get attention shape
                print(f"len(_att): {len(_att)}, _att[0].shape: {_att[0].shape}, _att[1].shape: {_att[1].shape}")
                average_attentions[layer_idx] = np.mean(_att, axis=0)
            att_pickle_name = out_pickle_prefix + "_att.pickle"
            with open(att_pickle_name, 'wb') as f:
                pickle.dump(average_attentions, f)
    else:
        for csize in set(cycle_size):
            average_states = dict()
            for layer_idx, activations in states.items():
                # filter by cycle size
                activations = [activations[i] for i in range(len(activations)) if cycle_size[i] == csize]
                average_states[layer_idx] = np.mean(activations, axis=0)
                # print(average_states[layer_idx].shape)
            out_pickle_name = out_pickle_prefix + f"_{csize}.pickle"
            with open(out_pickle_name, 'wb') as f:
                pickle.dump(average_states, f)

            if save_attention:
                average_attentions = dict()
                for layer_idx, attentions in attentions.items():
                    # filter by cycle size
                    attentions = [attentions[i] for i in range(len(attentions)) if cycle_size[i] == csize]
                    average_attentions[layer_idx] = np.mean(attentions, axis=0)
                    print(f"average_states[layer_idx].shape: {average_states[layer_idx].shape}")

                att_pickle_name = out_pickle_prefix + f"_{csize}_att.pickle"
                with open(att_pickle_name, 'wb') as f:
                    pickle.dump(average_attentions, f)

if __name__ == "__main__":
    typer.run(main)
