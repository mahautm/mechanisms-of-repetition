import torch
from torch.amp import autocast
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

class HookedModel(torch.nn.Module):
    def __init__(self, model, layer=None):
        super().__init__()
        self.model = model
        self.attn_outputs = []
        self.hooks = []
        self._register_attention_hooks(layer)

    def _register_attention_hooks(self, layer=None):
        # Register hooks on all attention layers
        for name, module in self.model.named_modules():
            if hasattr(module, 'attention') and (layer is None or layer == name):
                hook = module.attention.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
            elif hasattr(module, 'self_attn') and (layer is None or layer in name):
                hook = module.self_attn.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def _hook_fn(self, name):
        def fn(module, input, output):
            self.attn_outputs.append((name, output))
        return fn

    def clear(self):
        self.attn_outputs.clear()

    def forward(self, *args, **kwargs):
        self.clear()
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.clear()
        return self.model.generate(*args, **kwargs)
    
    def add_hooks(self, hooks_names):
        """
        Add hooks to the model for the specified layers.
        Silent failure if the layer does not exist. Nothing is added.
        """
        for name, module in self.model.named_modules():
            if name in hooks_names:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

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

def get_contrast(hooked_model, tokenizer, toked_input, expected_next, lens=None):
    acts_cyc = {}
    next_highest = {}

    try:
        o = hooked_model(**toked_input).logits
    except torch.cuda.OutOfMemoryError as e:
        # performance will take a hit here...
        print(f"OutOfMemoryError on CUDA, switching to CPU")
        hooked_model.to('cpu')
        toked_input = toked_input.to('cpu')
        with autocast(device_type='cpu', dtype=torch.float):
            o = hooked_model(toked_input).logits
        hooked_model.to('cuda')
        o = o.to('cuda')
            
    highest_toks = o[:, -1, :].argmax(-1).cpu()
    # masking out mismatches
    mask = highest_toks == torch.tensor(expected_next).cpu()
    # debug the error cases

    mismatches = torch.where(~mask)[0]
    if len(mismatches) > 0:
        print(f"Found {len(mismatches)} mismatches in the expected next token: {mismatches.tolist()}")
        for i in mismatches:
            print(f"Mismatch at index {i}: expected {expected_next[i]}, got {highest_toks[i].item()}")
            print(f"Input sentence: {tokenizer.decode(toked_input['input_ids'][i])}")
            print("Input tokens:", toked_input['input_ids'][i].tolist())
            print("Probability of expected next token:", torch.nn.functional.softmax(o[i, -1, :], dim=-1)[expected_next[i]].item())
            print("Probability of next highest token:", torch.nn.functional.softmax(o[i, -1, :], dim=-1)[highest_toks[i]].item())
            print("Difference:", torch.nn.functional.softmax(o[i, -1, :], dim=-1)[expected_next[i]].item() - torch.nn.functional.softmax(o[i, -1, :], dim=-1)[highest_toks[i]].item())
            print("Output shape:", o[i].shape)  # is the shape the one expected by the tokenizer?
            print("Toked input shape:", toked_input['input_ids'][i].shape)
            print("Was it in the top 5 tokens?", highest_toks[i].item() in torch.topk(o[i, -1, :], 5).indices.cpu().tolist())

    # get second highest token when the expected next token is not the highest
    second_highest_toks = o[:, -1, :].topk(2, dim=-1).indices[:, 1].cpu()
    highest_toks[mask] = second_highest_toks[mask]  # replace highest token with second highest where there is a mismatch

    for idx, v in enumerate(hooked_model.attn_outputs[-1]): # this could be done using generate, as we have a hook there anyway
        # print(k)
        k = hooked_model.hooks[idx].name
        if lens is not None:
            if torch.cuda.is_available():
                v=v[mask] # only keep the places where the expected next token was correct
                v = v.to('cuda')
                lens=lens.to('cuda')
            v = lens(v)
        # Apply detection functions if relevant

        unembedded = hooked_model.unembed(v)
        prob_expected = np.array([torch.nn.functional.softmax(unembedded[:, -1, head_idx, :].float(), dim=-1)[0,expected_next].item() for head_idx in range(unembedded.shape[2])])
        probs_next_highest = np.array([torch.nn.functional.softmax(unembedded[:, -1, head_idx, :].float(), dim=-1)[0,highest_toks].item() for head_idx in range(unembedded.shape[2])])
        if k not in acts_cyc:
            acts_cyc[k] = []
        acts_cyc[k].append(prob_expected)
        if k not in next_highest:
            next_highest[k] = []
        next_highest[k].append(probs_next_highest)


    return acts_cyc, next_highest

def extract_contrasts(text, hooked_model, tokenizer, lens=None, n_cycles=0, batch_size=1,max_length=256, max_new_tokens=100, no_head_analysis=False, layers=None):
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
        # print(f"decoded sample: {tokenizer.decode(tokenizer(sample, return_tensors='pt', padding=True, truncation=True, max_length=max_length)['input_ids'][0])}")
        toked= tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=max_length, padding_side="left")
        # print(f"toked input: {toked}")
        o1=hooked_model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens)
        if (o1[0][:toked["input_ids"].shape[1]] == toked["input_ids"][0]).all():
            plengths = toked["attention_mask"].sum(dim=1).tolist() # prompt_lengths in the batch
        else:
            plengths = [0] * len(o1) # no prompt lengths if the input is not in the output
            warn(f"Sample {i} does not start with the prompt, this will lead to incorrect contrast extraction. Executing anyway.")
        reps = [detect_cycles(tok[plengths[j]:], return_index=True, pad_token_id=tokenizer.pad_token_id) for j, tok in enumerate(o1)] # [(cycle_tokens, cycle_size, cycle_count, cycle_start_index) for each sample in the batch]
        # print(f"Repetitions found: {[reps for rep in reps if rep[0] is not None]}")
        for j, rep in enumerate(reps):
            if rep[0] is not None:
                print(f"Sample {i}, tokeded input {toked['input_ids'][j].tolist()}: cycle {rep[0]}, size {rep[1]}, count {rep[2]}, start index {rep[3]}")
            else:
                print(f"Sample {i}, repetition {j}: no cycle found")

        natural_input = [o1[j][:plengths[j] + rep[3]].tolist() + rep[0]*n_cycles for j,rep in enumerate(reps) if rep[0] is not None] # the natural input is the prompt and whatever was generated before the cycle starts, plus the cycle repeated n_cycles times
        print(f"Natural input: {natural_input}")
        data_index.extend([i for j in range(len(reps)) if reps[j][0] is not None])
        rep_index.extend([reps[j][3] for j in range(len(reps)) if reps[j][0] is not None])
        icl_input = [rep[0]*n_cycles for rep in reps if rep[0] is not None]
        expected_next = [rep[0][0] for rep in reps if rep[0] is not None]

        if len(natural_input) == 0 or no_head_analysis:
            warn(f"Skipping sample {i} as no repetitions were found or no head analysis is requested")
            continue

        # PADDING (TODO: Matéo Check on both passes the attention mask correctly happen)
        max_len = max(len(seq) for seq in natural_input)
        natural_input = [(([tokenizer.pad_token_id] * (max_len - len(seq))) + seq, [0] * (max_len - len(seq)) + [1] * len(seq)) for seq in natural_input]
        input_ids, attention_mask = zip(*natural_input)
        natural_input = {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}

        # CONTRASTS
        # get the contrast between the expected next token and the next highest token
        prob_expected, probs_next_highest = get_contrast(hooked_model, tokenizer, natural_input, expected_next, lens=lens)
        for k in prob_expected.keys():
            if k not in acts_cyc:
                acts_cyc[k] = []
            acts_cyc[k].extend(prob_expected[k])
        for k in probs_next_highest.keys():
            if k not in next_highest:
                next_highest[k] = []
            next_highest[k].extend(probs_next_highest[k])   

        if len(icl_input) == 0 or no_head_analysis:
            warn(f"Skipping ICL input for sample {i} as no repetitions were found or no head analysis is requested")
            continue

        # ICL PADDING
        max_len = max(len(seq) for seq in icl_input)
        icl_input = [(([tokenizer.pad_token_id] * (max_len - len(seq))) + seq, [0] * (max_len - len(seq)) + [1] * len(seq)) for seq in icl_input]
        icl_input, icl_attention_mask = zip(*icl_input)
        icl_input = {'input_ids': torch.tensor(icl_input), 'attention_mask': torch.tensor(icl_attention_mask)}
        
        prob_expected, probs_next_highest = get_contrast(hooked_model, tokenizer, icl_input, expected_next, lens=lens)
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


    probed_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision) if not use_bfloat16 else \
        AutoModelForCausalLM.from_pretrained_no_processing(model_name, revision=revision, torch_dtype=torch.bfloat16)
    probed_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probed_model.config.pad_token_id = tokenizer.pad_token_id

    hooked_model = HookedModel(probed_model, layer=None if single_lens is None else f"gpt_neox.layers.{str(single_lens)}")
    print(f"Loaded model {model_name} with {len(hooked_model.hooks)} attention hooks")
    print(f"hooked attention outputs are {hooked_model.hooks}")

    # DATA --> part of the pile that we are not using for the repetition analyisis
    dataset = load_dataset("JeanKaddour/minipile")
    subset = dataset["train"].shuffle(seed=seed)
    subset = subset.select(range(0, n_samples)) # 
    # subset = subset.select(range(0, len(subset) - 10000)) # skip last 10k datapoints
    
    # get activations
    full_heatmap = [None]*max_layer_idx
    icl_full_heatmap = [None]*max_layer_idx

    if lens_path is not None:
        lens = torch.load(lens_paths[single_lens if single_lens is None else 0], weights_only=False, map_location="cpu")
    else:
        lens = None
    
    layers= list(range(max_layer_idx)) if single_lens is None else [single_lens]
    heatmap, cy_count, icl_heatmap, icl_cy_count, d_idx, r_idx = extract_contrasts(subset["text"], hooked_model, tokenizer, lens=lens, n_cycles=n_cycles, batch_size=batch_size, max_length=max_length, max_new_tokens=max_new_tokens, no_head_analysis=no_head_analysis, layers=layers)
    for l in layers:
        full_heatmap[l] = heatmap[0] if single_lens is None else heatmap[0][l]
        icl_full_heatmap[l] = icl_heatmap[0] if single_lens is None else icl_heatmap[0][l] if icl_heatmap is not None else None

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
