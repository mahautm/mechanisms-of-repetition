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
        self.unembed_module = self._identify_unembed_module()
        
    def _identify_unembed_module(self):
        # Identify the final linear layer used for unembedding
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'head'):
            return self.model.head
        elif hasattr(self.model, 'embed_out'):
            return self.model.embed_out # for gpt-neox
        else:
            raise ValueError("Model does not have a recognizable unembedding layer.")

    def _register_attention_hooks(self, layer=None, only_layers=None):
        # Register hooks only for specified layers (if any)
        for name, module in self.model.named_modules():
            if only_layers and name not in only_layers:
                continue
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
    def unembed(self, x):
        # applies the model's final linear layer to the input x
        return self.unembed_module(x)
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

    # Use generate to capture activations (hooked_model.attn_outputs will be filled)
    try:
        o = hooked_model.generate(**toked_input, do_sample=False, max_new_tokens=1)
        logits = hooked_model.model.lm_head(hooked_model.model(**toked_input).last_hidden_state) if hasattr(hooked_model.model, 'lm_head') else None
    except torch.cuda.OutOfMemoryError as e:
        print(f"OutOfMemoryError on CUDA, switching to CPU")
        hooked_model.to('cpu')
        toked_input = {k: v.to('cpu') for k, v in toked_input.items()}
        with autocast(device_type='cpu', dtype=torch.float):
            o = hooked_model.generate(**toked_input, do_sample=False, max_new_tokens=1)
        hooked_model.to('cuda')


    # Use logits from the last forward pass if available, else skip
    if logits is None:
        return acts_cyc, next_highest

    device = logits.device
    # Ensure expected_next is a tensor on the same device as logits
    if not torch.is_tensor(expected_next):
        expected_next_tensor = torch.tensor(expected_next, device=device)
    else:
        expected_next_tensor = expected_next.to(device)
    highest_toks = logits[:, -1, :].argmax(-1)
    # Ensure highest_toks is on the same device as expected_next_tensor
    highest_toks = highest_toks.to(device)
    mask = highest_toks == expected_next_tensor
    mismatches = torch.where(~mask)[0]
    if len(mismatches) > 0:
        print(f"Found {len(mismatches)} mismatches in the expected next token: {mismatches.tolist()}")
        for i in mismatches:
            print(f"Mismatch at index {i}: expected {expected_next_tensor[i].item()}, got {highest_toks[i].item()}")
            print(f"Input sentence: {tokenizer.decode(toked_input['input_ids'][i])}")
            print("Input tokens:", toked_input['input_ids'][i].tolist())
            print("Probability of expected next token:", torch.nn.functional.softmax(logits[i, -1, :], dim=-1)[expected_next_tensor[i]].item())
            print("Probability of next highest token:", torch.nn.functional.softmax(logits[i, -1, :], dim=-1)[highest_toks[i]].item())
            print("Difference:", torch.nn.functional.softmax(logits[i, -1, :], dim=-1)[expected_next_tensor[i]].item() - torch.nn.functional.softmax(logits[i, -1, :], dim=-1)[highest_toks[i]].item())
            print("Output shape:", logits[i].shape)
            print("Toked input shape:", toked_input['input_ids'][i].shape)
            print("Was it in the top 5 tokens?", highest_toks[i].item() in torch.topk(logits[i, -1, :], 5).indices.cpu().tolist())

    # get second highest token when the expected next token is not the highest
    second_highest_toks = logits[:, -1, :].topk(2, dim=-1).indices[:, 1]
    highest_toks[mask] = second_highest_toks[mask]

    # Only process if attn_outputs is not empty
    if not hooked_model.attn_outputs:
        return acts_cyc, next_highest

    for idx, v in enumerate(hooked_model.attn_outputs[-1][1]):
        k = hooked_model.attn_outputs[-1][0] + "_head_" + str(idx)
        print(k, v.shape)
        if lens is not None:
            if torch.cuda.is_available():
                v = v[mask]
                v = v.to('cuda')
                lens = lens.to('cuda')
            v = lens(v)
            print(f"After lens, v shape: {v.shape}")
        hooked_model.unembed_module.to('cuda' if torch.cuda.is_available() else 'cpu')
        unembedded = hooked_model.unembed(v)
        if unembedded.shape[0] == 0:
            continue
        prob_expected = np.array([torch.nn.functional.softmax(unembedded[:, head_idx, :].float(), dim=-1)[0, expected_next].item() for head_idx in range(unembedded.shape[2])])
        probs_next_highest = np.array([torch.nn.functional.softmax(unembedded[:, head_idx, :].float(), dim=-1)[0, highest_toks].item() for head_idx in range(unembedded.shape[2])])
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
    # Pre-tokenize all text in advance for speed
    pretokenized = [tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length, padding_side="left") for t in text]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = min(batch_size, len(text))
    total_natural = 0
    total_icl = 0
    for i in tqdm(range(0, len(text), batch_size), desc="analysing samples", total=len(text)//batch_size):
        batch = pretokenized[i:i+batch_size]
        input_ids = torch.cat([b['input_ids'] for b in batch], dim=0)
        attention_mask = torch.cat([b['attention_mask'] for b in batch], dim=0)
        toked = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.float):
            o1 = hooked_model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens)
        # Cycle detection and contrast extraction logic (vectorized for batch)
    # 1. Find prompt lengths
    plengths = toked["attention_mask"].sum(dim=1).tolist()
    # 2. Ensure o1 and all tensors are on CPU for detect_cycles (if needed)
    o1_cpu = [o1[j].detach().cpu() if o1[j].device.type != 'cpu' else o1[j] for j in range(len(o1))]
    # 3. Detect cycles for each sample in the batch
    reps = [detect_cycles(o1_cpu[j][plengths[j]:], return_index=True, pad_token_id=tokenizer.pad_token_id) for j in range(len(o1_cpu))]
    # 4. Build natural and ICL inputs
    natural_input = [o1_cpu[j][:plengths[j] + rep[3]].tolist() + rep[0]*n_cycles for j, rep in enumerate(reps) if rep[0] is not None]
    icl_input = [rep[0]*n_cycles for rep in reps if rep[0] is not None]
    expected_next = [rep[0][0] for rep in reps if rep[0] is not None]
    data_index.extend([i + j for j in range(len(reps)) if reps[j][0] is not None])
    rep_index.extend([reps[j][3] for j in range(len(reps)) if reps[j][0] is not None])
    total_natural += len(natural_input)
    total_icl += len(icl_input)
    if len(natural_input) == 0 or no_head_analysis:
        warn(f"Skipping sample {i} as no repetitions were found or no head analysis is requested")
        # continue
    # PADDING for natural_input
    max_len = max(len(seq) for seq in natural_input)
    natural_input_padded = [(([tokenizer.pad_token_id] * (max_len - len(seq))) + seq, [0] * (max_len - len(seq)) + [1] * len(seq)) for seq in natural_input]
    input_ids, attention_mask = zip(*natural_input_padded)
    input_ids_tensor = torch.tensor(input_ids, device=device)
    attention_mask_tensor = torch.tensor(attention_mask, device=device)
    natural_input_dict = {'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor}
    prob_expected, probs_next_highest = get_contrast(hooked_model, tokenizer, natural_input_dict, expected_next, lens=lens)
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
        # continue
    # PADDING for icl_input
    max_len = max(len(seq) for seq in icl_input)
    icl_input_padded = [(([tokenizer.pad_token_id] * (max_len - len(seq))) + seq, [0] * (max_len - len(seq)) + [1] * len(seq)) for seq in icl_input]
    icl_input_ids, icl_attention_mask = zip(*icl_input_padded)
    icl_input_ids_tensor = torch.tensor(icl_input_ids)
    icl_attention_mask_tensor = torch.tensor(icl_attention_mask)
    # Ensure all tensors are on the same device
    icl_input_ids_tensor = icl_input_ids_tensor.to(device)
    icl_attention_mask_tensor = icl_attention_mask_tensor.to(device)
    # expected_next must be a tensor on the same device as input_ids
    expected_next_tensor = torch.tensor(expected_next, device=device)
    icl_input_dict = {'input_ids': icl_input_ids_tensor, 'attention_mask': icl_attention_mask_tensor}
    prob_expected, probs_next_highest = get_contrast(hooked_model, tokenizer, icl_input_dict, expected_next_tensor, lens=lens)
    for k in prob_expected.keys():
        if k not in icl_acts:
            icl_acts[k] = []
        icl_acts[k].extend(prob_expected[k])
    for k in probs_next_highest.keys():
        if k not in icl_next_highest:
            icl_next_highest[k] = []
        icl_next_highest[k].extend(probs_next_highest[k])


    # heatmap
    if acts_cyc and list(acts_cyc.keys()):
        first_key = list(acts_cyc.keys())[0]
        if acts_cyc[first_key]:
            heatmap = np.zeros((len(acts_cyc), len(acts_cyc[first_key][0])))
            for i, (k, v) in enumerate(acts_cyc.items()):
                c_acts = np.mean(v, axis=0)
                next_acts = np.mean(next_highest[k], axis=0)
                heatmap[i] = c_acts - next_acts
        else:
            heatmap = np.zeros((0, 0))
    else:
        heatmap = np.zeros((0, 0))

    if icl_acts and list(icl_acts.keys()):
        first_key = list(icl_acts.keys())[0]
        if icl_acts[first_key]:
            icl_heatmap = np.zeros((len(icl_acts), len(icl_acts[first_key][0])))
            for i, (k, v) in enumerate(icl_acts.items()):
                c_acts = np.mean(v, axis=0)
                next_acts = np.mean(icl_next_highest[k], axis=0)
                icl_heatmap[i] = c_acts - next_acts
        else:
            icl_heatmap = np.zeros((0, 0))
    else:
        icl_heatmap = None

    return heatmap, total_natural, icl_heatmap, total_icl, data_index, rep_index

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

    hooked_model = HookedModel(probed_model, layer=None if single_lens is None else f"gpt_neox.layers.{str(single_lens)}.attention")
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
