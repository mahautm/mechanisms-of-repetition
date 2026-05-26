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
from parrots.aa_fortu.aa_fortu_train_lens import Lens # this is what's loaded in the lens - we need it even if we don't call it
from datasets import load_dataset
from parrots.cycle_detection import detect_cycles
from warnings import warn

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

def get_contrast(hooked_model, toked_input, expected_next, cache, layers=None, lens=None):
    # CANNOT DEAL WITH BATCHES HERE TODO!!
    acts_cyc = {}
    next_highest = {}

    try:
        # with autocast(device_type='cuda', dtype=torch.bfloat16):
        # print("toked_input", toked_input)
        o = hooked_model(toked_input, stop_at_layer=max(layers) if layers is not None else None)
    except torch.cuda.OutOfMemoryError as e:
        # performance will take a hit here...
        print(f"OutOfMemoryError on CUDA, switching to CPU")
        hooked_model.to('cpu')
        toked_input = toked_input.to('cpu')
        with autocast(device_type='cpu', dtype=torch.float):
            o = hooked_model(toked_input, stop_at_layer=max(layers) if layers is not None else None)
        hooked_model.to('cuda')
        o = o.to('cuda')
            
    highest_toks = o[:, -1, :].argmax(-1).cpu()
    # print(f"highest token: {highest_toks}")
    # print(f"expected next token: {expected_next}")
    # input("Press Enter to continue...")
    if torch.equal(highest_toks, torch.tensor(expected_next)):
        highest_toks = torch.topk(o[:, -1, :], 2)
        highest_toks=highest_toks.indices[:, 1].item()
    elif not expected_next in torch.topk(o[:, -1, :], 5).indices.cpu().tolist():
        # maybe it's in the top 5, and there is very little difference?
        warn(f"Expected next token {expected_next} is not in the top 5 highest tokens {highest_toks} ")
        print(f"Input sentence: {hooked_model.tokenizer.decode(toked_input[0])}")
        print("Input tokens:", toked_input)
        # print("Output tokens:", o[0])
        print("Highest tokens:", highest_toks)
        print("Expected next token:", expected_next)
        print("probability of expected next token:", torch.nn.functional.softmax(o[:, -1, :], dim=-1)[0, expected_next].item())
        print("probability of next highest token:", torch.nn.functional.softmax(o[:, -1, :], dim=-1)[0, highest_toks].item())
        print("Difference:", torch.nn.functional.softmax(o[:, -1, :], dim=-1)[0, expected_next].item() - torch.nn.functional.softmax(o[:, -1, :], dim=-1)[0, highest_toks].item())
        print("output shape:", o.shape) # is the shape the one expected by the tokenizer?
        print("toked_input shape:", toked_input.shape)

        return {}, {}
    for k, v in cache.items():
        # print(k)
        if lens is not None:
            # with autocast(device_type='cuda', dtype=torch.bfloat16):
            if torch.cuda.is_available():
                v = v.to('cuda')
                lens=lens.to('cuda')
            v = lens(v)
        # Apply detection functions if relevant

        unembedded = hooked_model.unembed(v)
        # print(f"unembedded shape: {unembedded.shape}")
        prob_expected = np.array([torch.nn.functional.softmax(unembedded[:, -1, head_idx, :].float(), dim=-1)[0,expected_next].item() for head_idx in range(unembedded.shape[2])])
        probs_next_highest = np.array([torch.nn.functional.softmax(unembedded[:, -1, head_idx, :].float(), dim=-1)[0,highest_toks].item() for head_idx in range(unembedded.shape[2])])
        # print(f"prob_expected: {prob_expected}")
        # print(f"probs_next_highest: {probs_next_highest}")
        if k not in acts_cyc:
            acts_cyc[k] = []
        acts_cyc[k].append(prob_expected)
        if k not in next_highest:
            next_highest[k] = []
        next_highest[k].append(probs_next_highest)


    return acts_cyc, next_highest

def extract_contrasts(text, hooked_model, tokenizer, cache, lens=None, n_cycles=0, batch_size=1,max_length=256, max_new_tokens=100, no_head_analysis=False, layers=None):
    # attention head activations
    acts_cyc = {}
    next_highest = {}

    icl_acts = {}
    icl_next_highest = {}

    # ROUND 1 - for unexpected loops
    # for each sample
    data_index=[]
    rep_index=[]
    for i in tqdm(range(0, len(text), batch_size), desc="analysing samples", total=len(text)//batch_size):
        sample = text[i:i+batch_size]
        toked= tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        o1=hooked_model.generate(toked["input_ids"], do_sample=False, max_new_tokens=max_new_tokens)
        # are there repetitions?
        reps = [detect_cycles(tok[toked["input_ids"].shape[1]:], return_index=True) for tok in o1]

        ## DEBUGGING
        # print(f"Sample {i} - {tokenizer.decode(o1[0][:toked['input_ids'].shape[1]])} - {len(reps)} repetitions found")
        # print("---------------------")
        # print("----------------------")
        # print(tokenizer.decode(o1[0][toked["input_ids"].shape[1]:]))
        # print(o1[0][toked["input_ids"].shape[1]:])
        # print(f"cycle: {reps[0][0]}, cycle_size: {reps[0][1]}, cycle_count: {reps[0][2]}, cycle_start_index: {reps[0][3]}")
        # input(tokenizer.decode(reps[0][0]))

        
        natural_input = [o1[j][:toked["input_ids"].shape[1] + rep[3]].tolist() + rep[0]*n_cycles for j,rep in enumerate(reps) if rep[0] is not None]
        data_index.extend([i for j in range(len(reps)) if reps[j][0] is not None])
        rep_index.extend([reps[j][3] for j in range(len(reps)) if reps[j][0] is not None])
        icl_input = [rep[0]*n_cycles for rep in reps if rep[0] is not None]
        expected_next = [rep[0][0] for rep in reps if rep[0] is not None]

        # DEBUGGING
        print(f"natural_input: {natural_input}")
        print(f"natural sentence: {tokenizer.decode(natural_input[0]) if len(natural_input) > 0 else 'None'}")
        print(f"icl_input: {icl_input}")
        print(f"icl sentence: {tokenizer.decode(icl_input[0]) if len(icl_input) > 0 else 'None'}")
        print(f"expected_next: {expected_next}")

        if len(natural_input) == 0 or no_head_analysis:
            warn(f"Skipping sample {i} as no repetitions were found or no head analysis is requested")
            continue

        # get the activations
        torch.cuda.empty_cache()
        prob_expected, probs_next_highest = get_contrast(hooked_model, torch.tensor(natural_input), expected_next, cache, lens=lens, layers=layers)
        for k in prob_expected.keys():
            if k not in acts_cyc:
                acts_cyc[k] = []
            acts_cyc[k].extend(prob_expected[k])
        for k in probs_next_highest.keys():
            if k not in next_highest:
                next_highest[k] = []
            next_highest[k].extend(probs_next_highest[k])   

        if len(icl_input[0]) == 0:
            continue
        prob_expected, probs_next_highest = get_contrast(hooked_model, torch.tensor(icl_input), expected_next, cache, lens=lens, layers=layers)
        for k in prob_expected.keys():
            if k not in icl_acts:
                icl_acts[k] = []
            icl_acts[k].extend(prob_expected[k])
        for k in probs_next_highest.keys():
            if k not in icl_next_highest:
                icl_next_highest[k] = []
            icl_next_highest[k].extend(probs_next_highest[k])


    # heatmap
    heatmap = np.zeros((len(acts_cyc), len(acts_cyc[list(acts_cyc.keys())[0]][0])))
    for i, (k, v) in enumerate(acts_cyc.items()):
        c_acts = np.mean(v, axis=0)
        next_acts = np.mean(next_highest[k], axis=0)
        heatmap[i] = c_acts - next_acts
    
    icl_heatmap = np.zeros((len(icl_acts), len(icl_acts[list(icl_acts.keys())[0]][0]))) if len(icl_acts) > 0 else None

    for i, (k, v) in enumerate(icl_acts.items()):
        c_acts = np.mean(v, axis=0)
        next_acts = np.mean(icl_next_highest[k], axis=0)
        icl_heatmap[i] = c_acts - next_acts

    return heatmap, len([natural_input]), icl_heatmap, len([icl_input]), data_index, rep_index

def main(
    model_name:str="EleutherAI/pythia-1.4b",
    revision:str=None,
    # base_path:str="/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/perturbations",
    max_layer_idx:int=24,
    lens_path=None,
    single_lens:int=None,
    n_cycles:int=0,
    use_bfloat16:bool=False,
    seed:int=42,
    batch_size:int=1,
    max_length:int=256,
    max_new_tokens:int=100,
    no_head_analysis:bool=False,
    n_samples:int=5000
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
    subset = subset.select(range(0, n_samples)) # 
    # subset = subset.select(range(0, len(subset) - 10000)) # skip last 10k datapoints

    # get lenses
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
    
    # get activations
    full_heatmap = [None]*max_layer_idx
    icl_full_heatmap = [None]*max_layer_idx
    layer_idx = single_lens
    def hook_site(name: str):
        if name.endswith("hook_result"):
            block_idx = int(re.search(r"\.(\d+)\.", name).group(1))
            if block_idx == layer_idx:
                return True
        return False
    
    cache = hooked_model.add_caching_hooks(hook_site)

    if lens_path is not None:
        lens = torch.load(lens_paths[layer_idx if single_lens is None else 0], weights_only=False, map_location="cpu")
    else:
        lens = None
    
    heatmap, cy_count, icl_heatmap, icl_cy_count, d_idx, r_idx = extract_contrasts(subset["text"], hooked_model, tokenizer, cache, lens=lens, n_cycles=n_cycles, batch_size=batch_size, max_length=max_length, max_new_tokens=max_new_tokens, no_head_analysis=no_head_analysis, layers=[layer_idx])
    full_heatmap[layer_idx] = heatmap[0]
    icl_full_heatmap[layer_idx] = icl_heatmap[0] if icl_heatmap is not None else None
    if single_lens is not None:
        print(f"max_new_tokens: {max_new_tokens}")
        print(f"layer {single_lens} cycle count: {cy_count}")
        print(f"layer {single_lens} natural heatmap: {np.array2string(full_heatmap[single_lens], separator=',', max_line_width=np.inf)}")
        print(f"layer {single_lens} icl cycle count: {icl_cy_count}")
        print(f"layer {single_lens} icl heatmap: {np.array2string(icl_full_heatmap[single_lens], separator=',', max_line_width=np.inf) if icl_full_heatmap[single_lens] is not None else None}")
        print(f"layer {single_lens} data index: {d_idx}")
        print(f"layer {single_lens} repetition index: {r_idx}")

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

if __name__ == "__main__":
    typer.run(main)
