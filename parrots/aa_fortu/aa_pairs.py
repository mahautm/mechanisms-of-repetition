from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformer_lens import HookedTransformer
import re
import json
from tqdm import tqdm
import random
import pandas as pd

def hook_site(name: str):
    if name.endswith("hook_result") or name.endswith("hook_resid_mid") or name.endswith("hook_resid_pre"):
        block_idx = int(re.search(r"\.(\d+)\.", name).group(1))
        if block_idx >= 0:
            return True
    return False

def count_pair_head_act(text_inputs1, text_inputs2, hooked_model, cache, device, bz=2, thr=0.02):
    heads_active_freq = {f"{i}.{j}": 0 for i in range(0, hooked_model.cfg.n_layers) for j in range(hooked_model.cfg.n_heads)}
    # alternate 1:1 so that even are from one and odd are from the other
    assert len(text_inputs1) == len(text_inputs2), "inputs must have the same length"
    assert bz % 2 == 0, "batch size must be even"
    text_inputs = []
    for i in range(len(text_inputs1)):
        text_inputs.append(text_inputs1[i])
        text_inputs.append(text_inputs2[i])
    del text_inputs1
    del text_inputs2

    print("len of text", len(text_inputs))
    if not isinstance(text_inputs[0], str):
        text_inputs = torch.cat(text_inputs, dim=0)
    for i in tqdm(range(0, len(text_inputs), bz)):
        text_batch = text_inputs[i:i+bz]
        if isinstance(text_batch[0], str):
            tokens_batch = hooked_model.to_tokens(text_batch)
        else:
            tokens_batch = text_batch.to(device)
        hooked_model(tokens_batch, return_type=None)

        seq_lengths = torch.eq(tokens_batch, hooked_model.tokenizer.pad_token_id).int().argmax(-1) - 1
        arange_idx = torch.arange(tokens_batch.size(0), device=device)

        for j in range(0, hooked_model.cfg.n_layers):
            parts_vec = cache[f"blocks.{j}.attn.hook_result"][arange_idx, seq_lengths].transpose(0, 1)   # [batch, pos, head_index, d_model] -> [head_idx, batch, d_model]
            parts_vec = torch.cat([cache[f"blocks.{j}.hook_resid_pre"][arange_idx, seq_lengths].unsqueeze(0), parts_vec], dim=0)
            if f"blocks.{j}.hook_resid_mid" in cache:
                whole_vec = cache[f"blocks.{j}.hook_resid_mid"][arange_idx, seq_lengths].contiguous()   # [batch, d_model]
            else:
                # as pythia does not have hook_resid_mid, we need to calculate it
                whole_vec = parts_vec.sum(dim=0) + cache[f"blocks.{j}.hook_resid_pre"][arange_idx, seq_lengths].contiguous()   # [batch, d_model]

            temp_whole_vec = whole_vec.unsqueeze(0).expand_as(parts_vec)
            distance = torch.nn.functional.pairwise_distance(parts_vec, temp_whole_vec, p=1)

            whole_norm = torch.norm(whole_vec, p=1, dim=-1)
            proximity = (whole_norm.unsqueeze(0) - distance).clip(min=1e-5)

            proximity /= proximity.sum(dim=0, keepdim=True) # head_idx, batch

            proximity = proximity[1:].contiguous()
            # substract odds from evens
            proximity = proximity[1::2] - proximity[0::2]

            temp_freq = (proximity > thr).sum(dim=1)
            for k, n in enumerate(temp_freq.tolist()):
                heads_active_freq[f"{j}.{k}"] += n

            # Move tensors to CPU to free up GPU memory
            whole_vec.cpu()
            parts_vec.cpu()
            temp_whole_vec.cpu()
            distance.cpu()
            whole_norm.cpu()
            proximity.cpu()
            temp_freq.cpu()

    return heads_active_freq

if __name__ == "__main__":

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", add_bos_token=True)

    df = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results_with_cycles.csv")

    # deal with na in cycle by replacing with empty string
    df["cycle"] = df["cycle"].fillna("")
    no_cycle_inputs = df[df["cycle"] == ""]["to_send"].tolist()
    cycle_inputs = df[df["cycle"] != ""]["to_send"].tolist()

    # cycle end
    df["pre_cycle"] = df.apply(lambda x:(x["to_send"] + x['generated'].split(x["cycle"])[0]) if x["cycle"] != "" else x["to_send"], axis=1)
    dir_cycle_inputs = df[df["cycle"] != ""]["pre_cycle"].tolist()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    probed_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    probed_model.config.pad_token_id = tokenizer.pad_token_id

    hooked_model = HookedTransformer.from_pretrained(
            "facebook/opt-1.3b",
            hf_model=probed_model,
            tokenizer=tokenizer,
            n_embd=probed_model.config.hidden_size,
            n_layer=probed_model.config.num_hidden_layers,
            n_head=probed_model.config.num_attention_heads,
            vocab_size=probed_model.config.vocab_size,
            n_ctx=probed_model.config.max_position_embeddings,
    )

    hooked_model.eval()
    del probed_model
    hooked_model.set_use_attn_result(True)
    cache = hooked_model.add_caching_hooks(hook_site)
    cycle_heads_active_freq = count_head_act(cycle_inputs, hooked_model, cache, device)
    heads_active_freq = count_head_act(no_cycle_inputs, hooked_model, cache, device)
    # print showing 10 biggest differences
    biggest_diff = {}
    for k, v in heads_active_freq.items():
        biggest_diff[k] = cycle_heads_active_freq[k] - v
    biggest_diff = sorted(biggest_diff.items(), key=lambda x: x[1], reverse=True)
    print(biggest_diff[:10])

    dir_cycle_heads_active_freq = count_head_act(dir_cycle_inputs, hooked_model, cache, device)
    # dir
    biggest_diff = {}
    for k, v in heads_active_freq.items():
        biggest_diff[k] = dir_cycle_heads_active_freq[k] - v
    biggest_diff = sorted(biggest_diff.items(), key=lambda x: x[1], reverse=True)
    print(biggest_diff[:10])

    # save results
    with open("heads_active_freq.json", "w") as f:
        json.dump(heads_active_freq, f)
    with open("cycle_heads_active_freq.json", "w") as f:
        json.dump(cycle_heads_active_freq, f)
    with open("dir_cycle_heads_active_freq.json", "w") as f:
        json.dump(dir_cycle_heads_active_freq, f)

# This needs to be reran to look at the difference between "direct cycle" where the cycle happens directly after the prompt
# and indirect cycle where we IGNORE the part of the generation that is not a cycle. 
# Results already show differences with indirect cycle
# here we hope instead they'll be less different