from parrots.attention_analysis import count_head_act, hook_site
from parrots.aa_pairs import count_pair_head_act
from parrots.random_rep import data_generation
from parrots.cycle_detection import detect_cycles
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from pathlib import Path
import torch
import json
import typer
import time
import random
from threading import Lock
import pandas as pd


def get_non_repeating_generation(prompt, model, tokenizer, device, max_length=100):
    # generate a response
    for _ in range(1000):
        generated = model.generate(
            prompt.unsqueeze(0).to(device),
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )
        _, _, cycle_count=detect_cycles(generated[0])
        if cycle_count == 0:
            return generated[0]
        # check if the prompt is repeated

    return None

def main(
    cycle_size: int,
    idx: int,
    dataset_size: int,
    batch_size: int,
    model_name: str,
    save_path: str="",
    revision:str=None,
    cache_dir:str=None,
    use_bnb: bool=False,
    n_devices: int=1,
):
    # idx is the index of the token in the cycle we want to analyze
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    tokenizer_lock = Lock()

    def load_tokenizer(model_name, revision, cache_dir):
        with tokenizer_lock:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        return tokenizer

    tokenizer = load_tokenizer(model_name, revision, cache_dir)
    probed_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, load_in_4bit=use_bnb)
    hooked_model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=probed_model,
            tokenizer=tokenizer,
            n_embd=probed_model.config.hidden_size,
            n_layer=probed_model.config.num_hidden_layers,
            n_head=probed_model.config.num_attention_heads,
            vocab_size=probed_model.config.vocab_size,
            n_ctx=probed_model.config.max_position_embeddings,
            n_devices=n_devices,
    )
    hooked_model.to(device)
    hooked_model.eval()
    hooked_model.set_use_attn_result(True) 
    del probed_model
    torch.cuda.empty_cache()

    df = pd.read_csv("/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b_human_lama_parrots_list_v1_sf/slot_filling_results_with_cycles.csv")
    # df=pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results_with_cycles.csv")
    df = df[df["cycle"] != "  "]
    df = df[df["cycle_size"] == cycle_size]
    df = df.sample(1000 if len(df) > 1000 else len(df))
    # df = df[df["cycle_size"] <= 3]
    # # sample 50 rows from each cycle size
    # df = df.groupby("cycle_size").apply(lambda x: x.sample(min(100, len(x)))).reset_index(drop=True)
    
    df["generated_from_cycle_start"] = df.apply(lambda row: row["generated"][row["generated"].find(str(row["cycle"])):] if row["cycle_size"] > 0 else row["generated"], axis=1)
    df["toked_gen_cycle"] = df["generated_from_cycle_start"].apply(lambda x: tokenizer([x], return_tensors="pt")["input_ids"][0].cpu())
    df["toked_all_gen"] = df["generated"].apply(lambda x: tokenizer([x], return_tensors="pt")["input_ids"][0].cpu())
    df["cycle_start_index"] = df.apply(lambda row: len(row["toked_all_gen"]) - len(row["toked_gen_cycle"]), axis=1)

    cycle_start_index = df["cycle_start_index"].tolist()
    cycle_data = df["toked_all_gen"].tolist()
    nocycle_data = df.apply(lambda row: get_non_repeating_generation(
            row["toked_all_gen"][:row["cycle_start_index"]if row["cycle_start_index"] > 0 else 1],
            hooked_model,
            tokenizer,
            device
        ), axis=1).tolist()

    part_cycle_data = [x[:cycle_start_index[i] + idx].unsqueeze(0) for i,x in enumerate(cycle_data)]
    part_no_cycle_data = [torch.cat((x[:cycle_start_index[i]], nocycle_data[i][cycle_start_index[i]:idx]), dim=0).unsqueeze(0) for i, x in enumerate(cycle_data)]
    eos_token_id = tokenizer.eos_token_id
    # pad everything to the same size
    max_size = max([x.shape[1] for x in part_cycle_data])
    max_size = max(max_size, max([x.shape[1] for x in part_no_cycle_data]))
    part_cycle_data = [torch.cat((x, torch.tensor([eos_token_id] * (max_size - x.shape[1]), dtype=torch.long).unsqueeze(0)), dim=1) for x in part_cycle_data]
    part_no_cycle_data = [torch.cat((x, torch.tensor([eos_token_id] * (max_size - x.shape[1]), dtype=torch.long).unsqueeze(0)), dim=1) for x in part_no_cycle_data]

    cache = hooked_model.add_caching_hooks(hook_site)
    att_cycles = count_pair_head_act(part_cycle_data, part_no_cycle_data, hooked_model, cache, device, thr=0.01)
    # att_cycles = count_head_act(part_cycle_data, hooked_model, cache, device, thr=0.01)
    # att_random = count_head_act(random_data, hooked_model, cache, device, thr=0.01)
    save_path = Path(save_path)
    # pars=f"{n_cycles}_{cycle_size}_{idx}"
    pars=f"{cycle_size}_{idx}"
    if revision:
        pars+=f"_{revision}"

    # add all parameters to json under the key params
    att_cycles["params"] = {
        "cycle_size": int(cycle_size),
        "dataset_size": int(dataset_size),
        "batch_size": int(batch_size),
        "model_name": model_name,
        "revision": revision,
        "idx": int(idx)
    }
    # att_random["params"] = {"n_cycles": n_cycles, "cycle_size": cycle_size, "dataset_size": dataset_size, "batch_size": batch_size, "model_name": model_name, "revision":revision}
    print(att_cycles)
    with open(save_path / f"{pars}_att_cycles.json", "w") as f:
        json.dump(att_cycles, f)
    # with open(save_path / f"{pars}_att_random.json", "w") as f:
    #     json.dump(att_random, f)
    
if __name__ == "__main__":
    typer.run(main)