from parrots.attention_analysis import count_head_act, hook_site
from parrots.aa_pairs import count_pair_head_act
from parrots.random_rep import data_generation
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from pathlib import Path
import torch
import json
import typer
import time
import random
from threading import Lock


def main(
    cycle_size: int,
    n_cycles: int,
    dataset_size: int,
    batch_size: int,
    model_name: str,
    save_path: str="",
    revision:str=None,
    cache_dir:str=None,
    use_bnb: bool=False,
    n_devices: int=1,
):
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

    cycle_data = data_generation(tokenizer, cycle_size, n_cycles, dataset_size, batch_size)
    cycle_data = list(cycle_data)
    nocycle_data = data_generation(tokenizer, cycle_size*n_cycles, 1, dataset_size, batch_size)
    nocycle_data = list(nocycle_data)
    # modify this to have activation pairs
    
    for idx in range(cycle_size):
        part_cycle_data = [x[:cycle_size + idx] for x in cycle_data]
        print(f"part_cycle_data: {part_cycle_data}")
        input()
        part_no_cycle_data = [torch.cat((x[:cycle_size], nocycle_data[i][cycle_size:cycle_size+idx]), dim=0) for i, x in enumerate(cycle_data)]
        cache = hooked_model.add_caching_hooks(hook_site)
        att_cycles = count_pair_head_act(part_cycle_data, part_no_cycle_data, hooked_model, cache, device, thr=0.01)
        # att_cycles = count_head_act(part_cycle_data, hooked_model, cache, device, thr=0.01)
        # att_random = count_head_act(random_data, hooked_model, cache, device, thr=0.1)
        save_path = Path(save_path)
        pars=f"{n_cycles}_{cycle_size}_{idx}"
        if revision:
            pars+=f"_{revision}"

        # add all parameters to json under the key params
        att_cycles["params"] = {"n_cycles": n_cycles, "cycle_size": cycle_size, "dataset_size": dataset_size, "batch_size": batch_size, "model_name": model_name, "revision":revision, "idx":idx}
        # att_random["params"] = {"n_cycles": n_cycles, "cycle_size": cycle_size, "dataset_size": dataset_size, "batch_size": batch_size, "model_name": model_name, "revision":revision}
        with open(save_path / f"{pars}_att_cycles.json", "w") as f:
            json.dump(att_cycles, f)
        # with open(save_path / f"{pars}_att_random.json", "w") as f:
        #     json.dump(att_random, f)
    
if __name__ == "__main__":
    typer.run(main)