import torch
from torch.amp import autocast
import torch.nn as nn
from transformer_lens import HookedTransformer
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from parrots.aa_fortu_train_lens import Lens

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
# class Lens(nn.Module):
#     def __init__(self, embed_size):
#         super(Lens, self).__init__()
#         self.lens = nn.Linear(embed_size, embed_size)
#         self.bias = nn.Parameter(torch.zeros(embed_size))
    
#     def forward(self, x):
#         return self.lens(x) + self.bias
        
def extract_contrasts(df, hooked_model, cache, icl_data=False, lens=None, n_cycles=1):
    # attention head activations
    acts_cyc = {}
    next_highest = {}

    # ROUND 1 - for unexpected loops
    # for each sample
    cycle_count = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # get the first token
        if icl_data:
            input_tok = torch.tensor(row["cycle"]).repeat(1+n_cycles).unsqueeze(0)
            if len(input_tok[0]) > df["toked_to_send"].apply(len).max():
                continue
            expected_next = row["cycle"][0]
        else:
            cycle_len= len(row["cycle"])
            if row["cycle_start_index"] + (cycle_len) * n_cycles + 1 >= len(row["toked_to_send"]):
                continue
            input_tok = torch.tensor(row["toked_to_send"][:row["cycle_start_index"] + (cycle_len) * n_cycles + 1]).unsqueeze(0)
            expected_next = torch.tensor(row["toked_to_send"][row["cycle_start_index"] + 1:row["cycle_start_index"] + 2]).item()
        
        # get the activations
        torch.cuda.empty_cache()
        try:
            with autocast(device_type='cuda'): #, dtype=torch.bfloat16):
                o = hooked_model(input_tok)
        except torch.cuda.OutOfMemoryError as e:
            # performance will take a hit here...
            print(f"OutOfMemoryError on CUDA, switching to CPU on sample {i}")
            hooked_model.to('cpu')
            input_tok = input_tok.to('cpu')
            with autocast(device_type='cpu', dtype=torch.float):
                o = hooked_model(input_tok)
            hooked_model.to('cuda')
            o = o.to('cuda')
                
        highest_tok = o[0, -1, :].argmax().item()
        if highest_tok == expected_next:
            cycle_count += 1
            highest_tok = torch.topk(o[0, -1, :], 2).indices[1].item()
        else:
            continue
        for k, v in cache.items():
            if lens is not None:
                # with autocast(device_type='cuda', dtype=torch.bfloat16):
                v=v.to('cuda', dtype=torch.float32)
                lens=lens.to('cuda', dtype=torch.float32)
                v = lens(v)
            unembedded = hooked_model.unembed(v)
            prob_expected = torch.nn.functional.softmax(unembedded[0, -1, :].float(), dim=-1)[:,expected_next].cpu().detach().numpy()
            probs_next_highest = np.array([torch.nn.functional.softmax(unembedded[0, -1, head_idx, :].float(), dim=-1)[highest_tok].item() for head_idx in range(unembedded.shape[2])])
            if k not in acts_cyc:
                acts_cyc[k] = []
            acts_cyc[k].append(prob_expected)

            if k not in next_highest:
                next_highest[k] = []
            next_highest[k].append(probs_next_highest)

    # heatmap
    heatmap = np.zeros((len(acts_cyc), len(acts_cyc[list(acts_cyc.keys())[0]][0])))
    for i, (k, v) in enumerate(acts_cyc.items()):
        c_acts = np.mean(v, axis=0)
        next_acts = np.mean(next_highest[k], axis=0)
        heatmap[i] = c_acts - next_acts

    return heatmap, cycle_count/len(df)

def main(
    model_name:str="EleutherAI/pythia-1.4b",
    revision:str=None,
    base_path:str="/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations",
    max_layer_idx:int=24,
    lens_path:str=None,
    single_lens:int=None,
    n_cycles:int=1,
    sample:int=None,
    use_bfloat16:bool=False
    ):

    # LOAD model using lens
    probed_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision) #if not use_bfloat16 else \
        # AutoModelForCausalLM.from_pretrained_no_processing(model_name, revision=revision, torch_dtype=torch.bfloat16)
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

    # DATA (packing, to be removed?)
    cycle_size = 3
    df = pd.DataFrame()
    for f in Path(base_path).glob(f"cycle_{cycle_size}_results_*.csv"):
        df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
    df = df[df["cycle_size"] != 0]
    df = df[df["cycle_size"] != 1]
    df = df[df["cycle_count"] != 0]
    df = df[df["cycle_count"] != 1]
    if sample is not None:
        df = df.sample(sample)


    df["toked_input"] = df["toked_input"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])
    df["cycle"] = df["cycle"].apply(lambda x: np.array(pd.eval(x)) if isinstance(x, str) else [])

    df["toked_generated"] = df["generated"].progress_apply(lambda x: tokenizer(x, return_tensors="pt")["input_ids"][0].cpu().numpy() if x != "" else [])
    df["toked_to_send"] = df.apply(lambda row: np.concatenate([row["toked_input"], row["toked_generated"]]), axis=1)
    df["cycle_start_index"] = df.progress_apply(lambda row: find_seq_idx(row["cycle"], row["toked_to_send"]), axis=1)
    df= df[df["cycle_start_index"] != -1]
    df= df[df["cycle_start_index"] != 0]
    print("Number of samples:", len(df))

    if lens_path is not None:
        lens_paths = list(Path(lens_path).glob("*.pth"))
        if len(lens_paths) == 0 or lens_paths is None:
            raise ValueError(f"No lens found in {lens_path}")
        
        lens_paths.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))
        print(f"Found {len(lens_paths)} lenses in {lens_path} ({[x.stem for x in lens_paths]})")
        if single_lens is not None:
            lens_paths = [lens_paths[single_lens]]
            print(f"Keeping only lens {lens_paths[0]}")
        else:
            assert len(lens_paths) == max_layer_idx, f"Expected {max_layer_idx} lenses, one for each layer, instead got {len(lens_paths)}"
    
    full_heatmap = [None]*max_layer_idx
    layers = range(max_layer_idx) if single_lens is None else [single_lens]
    for layer_idx in layers:
        def hook_site(name: str):
            if name.endswith("hook_result"):
                block_idx = int(re.search(r"\.(\d+)\.", name).group(1))
                if block_idx == layer_idx:
                    return True
            return False
        cache = hooked_model.add_caching_hooks(hook_site)

        if lens_path is not None:
            lens = torch.load(lens_paths[layer_idx if single_lens is None else 0], weights_only=False)
        else:
            lens = None
        heatmap, cy_count = extract_contrasts(df, hooked_model, cache, lens=lens, n_cycles=n_cycles)
        full_heatmap[layer_idx] = heatmap[0]
    if single_lens is not None:
        print(f"layer {single_lens} cycle count: {cy_count}")
        print(f"layer {single_lens} natural heatmap: {np.array2string(full_heatmap[single_lens], separator=',', max_line_width=np.inf)}")
        if single_lens == max_layer_idx - 1:
            full_heatmap = plot_from_logs(base_path)
    if single_lens is None or single_lens == max_layer_idx - 1:
        # Plot and save heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(full_heatmap, cmap="viridis", cbar=True, vmin=0, vmax=0.2)
        plt.title(f"unexpected activation contrast Heatmap {n_cycles}")
        plt.suptitle(f"proportion of cycles: {cy_count}, cycle number: {n_cycles}")
        plt.xlabel("Attention Heads")
        plt.ylabel("Samples")
        plt.savefig(f"unexpected_lens_heatmap_contrast_{n_cycles}.png")
        plt.close()

    full_heatmap = [None]*max_layer_idx
    for layer_idx in layers:
        def hook_site(name: str):
            if name.endswith("hook_result"):
                block_idx = int(re.search(r"\.(\d+)\.", name).group(1))
                if block_idx == layer_idx:
                    return True
            return False
        cache = hooked_model.add_caching_hooks(hook_site)

        if lens_path is not None:
            lens = torch.load(lens_paths[layer_idx if single_lens is None else 0], weights_only=False).to('cuda')
        else:
            lens = None
        heatmap, cy_count = extract_contrasts(df, hooked_model, cache, icl_data=True, lens=lens, n_cycles=n_cycles)
        full_heatmap[layer_idx] = heatmap[0]
    if single_lens is not None:
        print(f"layer {single_lens} cycle count: {cy_count}")
        print(f"layer {single_lens} icl heatmap: {np.array2string(full_heatmap[single_lens], separator=',', max_line_width=np.inf)}")
        if single_lens == max_layer_idx - 1:
            full_heatmap = plot_from_logs(base_path, heatmap_type="icl")

    if single_lens is None or single_lens == max_layer_idx - 1:
        # Plot and save heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(full_heatmap, cmap="viridis", cbar=True, vmin=0, vmax=0.2)
        plt.title(f"ICL activation contrast Heatmap")
        plt.suptitle(f"proportion of cycles: {cy_count}, cycle number: {n_cycles}")
        plt.xlabel("Attention Heads")
        plt.ylabel("Samples")
        plt.savefig(f"icl_lens_heatmap_contrast_{n_cycles}.png")
        plt.close()

def plot_from_logs(base_path, heatmap_type="natural"):
    # collect logs in base_path
    logs = []
    for f in Path(base_path).glob("*.out"):
        with open(f, "r") as file:
            logs.append(file.read())
    # Extract heatmap elements from logs
    heatmaps = []
    for log in logs:
        matches = re.findall(fr"layer (\d+) {heatmap_type} heatmap: (.+)", log)
        for layer, match in matches:
            heatmap = np.array(eval(match))
            heatmaps.append((int(layer), heatmap))
        heatmaps.sort(key=lambda x: x[0])
        heatmaps = [heatmap for _, heatmap in heatmaps]
    return heatmaps

if __name__ == "__main__":
    typer.run(main)
