import torch
from torch.amp import autocast
from transformer_lens import HookedTransformer
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer
import re
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from parrots.cycle_detection import detect_cycles
from dadapy import Data
import pandas as pd

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

def get_reps(tokens, hooked_model, cache, batch_size=1, layers=None):
    """ Get the activations for the given tokens using the hooked model.
    """
    reps = {}
    for i in tqdm(range(0, len(tokens)//batch_size, batch_size), desc="Getting activations", total=len(tokens)//batch_size, leave=False):
        toks = tokens[i*batch_size:(i+1)*batch_size] # list of lists of tokens
        # padding, with the pad token, then convert to tensor
        max_len = max(len(t) for t in toks)
        toks = [t + [hooked_model.tokenizer.pad_token_id] * (max_len - len(t)) for t in toks]
        toks = torch.tensor(toks)

        try:
            o = hooked_model(toks, stop_at_layer=max(layers) if layers is not None else None)
        except torch.cuda.OutOfMemoryError as e:
            # performance will take a hit here...
            print(f"OutOfMemoryError on CUDA, switching to CPU")
            hooked_model.to('cpu')
            toks = toks.to('cpu')
            with autocast(device_type='cpu', dtype=torch.float):
                o = hooked_model(toks, stop_at_layer=max(layers) if layers is not None else None)
            hooked_model.to('cuda')
            o = o.to('cuda')

        # get the activations
        for k, v in cache.items():
            if k not in reps:
                reps[k] = []
            print(f"Block {k} has shape {v.shape}")
            reps[k].extend(v.cpu().numpy().mean(axis=1))  # mean over sequence length

    return reps

def get_ID(reps, range_max=16):
    """ Compute the intrinsic dimension of the given activations.
    reps must be a dictionary showing as keys layers/blocks/heads and as values the corresponding activations.
    range_max is the maximum range for the intrinsic dimension computation, ID will be computed for each power of 2 up to range_max.
    """
    ids = {}
    
    for i, (k,v) in enumerate(reps.items()):
        # v has shape (batch_size, attention_heads, embedding_size)
        v= np.array(v)
        for head in range(v.shape[1]):
            # get the activations for the head
            head_activations = v[:, head, :]
            dada_data = Data(coordinates=head_activations)
            ids_scaling, _, _ = dada_data.return_id_scaling_gride(range_max=range_max, set_attr=True)
            ids[f"{k}_head_{head}"] = ids_scaling
    return ids

def extract_ID(text, hooked_model, tokenizer, cache, n_cycles=0, batch_size=1,max_length=256, max_new_tokens=100, minimum_id_batch_size=500, layers=None):
    assert minimum_id_batch_size <= len(text), f"minimum_id_batch_size ({minimum_id_batch_size}) must be less than or equal to the number of samples ({len(text)})"
    nat_ids = {}
    icl_ids = {}

    # ROUND 1 - for unexpected loops
    # for each sample
    data_index=[]
    rep_index=[]
    stacked_natural_input = []
    stacked_icl_input = []
    for i in tqdm(range(0, len(text), batch_size), desc="analysing samples", total=len(text)//batch_size):
        sample = text[i:i+batch_size]
        toked= tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        o1=hooked_model.generate(toked["input_ids"], do_sample=False, max_new_tokens=max_new_tokens)
        # are there repetitions?
        reps = [detect_cycles(tok[toked["input_ids"].shape[1]:], return_index=True) for tok in o1]

        
        natural_input = [o1[j][:toked["input_ids"].shape[1] + rep[3]].tolist() + rep[0].tolist()*n_cycles for j,rep in enumerate(reps) if rep[0] is not None]
        data_index.extend([i for j in range(len(reps)) if reps[j][0] is not None])
        rep_index.extend([reps[j][3] for j in range(len(reps)) if reps[j][0] is not None])
        icl_input = [rep[0].tolist()*n_cycles for rep in reps if rep[0] is not None]

        stacked_natural_input.extend(natural_input)
        stacked_icl_input.extend(icl_input)
        # memory management
        del toked, o1, reps, natural_input, icl_input
        torch.cuda.empty_cache()

    if len(stacked_natural_input) >= minimum_id_batch_size:
        # get the activations
        torch.cuda.empty_cache()
        reps = get_reps(stacked_natural_input, hooked_model, cache, batch_size=batch_size, layers=layers)
        id = get_ID(reps)
        nat_ids.update(id)

        if n_cycles > 0:
            reps_icl = get_reps(stacked_icl_input, hooked_model, cache, batch_size=batch_size, layers=layers)
            icl_id = get_ID(reps_icl)
            icl_ids.update(icl_id)
        else:
            icl_ids = None
    else:
        print(f"Not enough samples for ID computation, got {len(stacked_natural_input)} samples, minimum is {minimum_id_batch_size}")
        nat_ids = {}
        icl_ids = None



    print(f"natural ids: {nat_ids.keys()}")
    print(f"icl ids: {icl_ids.keys() if icl_ids is not None else 'None'}")


    return nat_ids, icl_ids

def main(
    model_name:str="EleutherAI/pythia-1.4b",
    revision:str=None,
    # base_path:str="/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations",
    max_layer_idx:int=24,
    layer_idx:list[int]=None,
    n_cycles:int=0,
    use_bfloat16:bool=False,
    seed:int=42,
    batch_size:int=16,
    max_length:int=16,
    max_new_tokens:int=512,
    minimum_id_batch_size:int=10,
    ):

    # LOAD model using lens
    probed_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision) if not use_bfloat16 else \
        AutoModelForCausalLM.from_pretrained_no_processing(model_name, revision=revision, torch_dtype=torch.bfloat16)
    probed_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probed_model.config.pad_token_id = tokenizer.pad_token_id

    hooked_model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=probed_model,
            tokenizer=tokenizer,
            n_embd=probed_model.config.hidden_size,
            n_layer=probed_model.config.num_hidden_layers,
            n_head=probed_model.config.num_attention_heads,
            vocab_size=probed_model.config.vocab_size,
            n_ctx=probed_model.config.max_position_embeddings,
            dtype=torch.bfloat16 if use_bfloat16 else None
    )

    hooked_model.eval()
    del probed_model
    hooked_model.set_use_attn_result(True)
    torch.set_grad_enabled(False)

    # DATA --> part of the pile that we are not using for the repetition analyisis
    dataset = load_dataset("JeanKaddour/minipile")
    subset = dataset["train"].shuffle(seed=seed)
    subset = subset.select(range(0, 1000)) # 
    # subset = subset.select(range(0, len(subset) - 10000)) # skip last 10k datapoints

    layer_idx = layer_idx if layer_idx is not None else list(range(max_layer_idx))
    def hook_site(name: str):
        if name.endswith("hook_result"):
            block_idx = int(re.search(r"\.(\d+)\.", name).group(1))
            if block_idx in layer_idx:
                return True
        return False
    
    cache = hooked_model.add_caching_hooks(hook_site)
    
    nat_ids, icl_ids = extract_ID(subset["text"], hooked_model, tokenizer, cache, n_cycles=n_cycles, batch_size=batch_size, max_length=max_length, max_new_tokens=max_new_tokens, minimum_id_batch_size=minimum_id_batch_size, layers=layer_idx)
    
    # print the ids - all of them without compression
    print("Natural IDs:")
    np.set_printoptions(threshold=np.inf)
    for k, v in nat_ids.items():
        print(f"Layer {k}: {np.array(v)}")

    if icl_ids is not None:
        print("ICL IDs:")
        for k, v in icl_ids.items():
            print(f"Layer {k}: {np.array(v)}")

    # make a graph of the IDs - in a nature plot - use the right font, right size, right colors, right ppi
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pd.DataFrame(nat_ids), dashes=False, markers=True)
    plt.title(f"Intrinsic Dimension for {model_name} - Natural IDs")
    plt.xlabel("Layer")
    plt.ylabel("Intrinsic Dimension")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{model_name}_nat_ids.png", dpi=300)
    if icl_ids is not None:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=pd.DataFrame(icl_ids), dashes=False, markers=True)
        plt.title(f"Intrinsic Dimension for {model_name} - ICL IDs")
        plt.xlabel("Layer")
        plt.ylabel("Intrinsic Dimension")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{model_name}_icl_ids.png", dpi=300)

if __name__ == "__main__":
    typer.run(main)
