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

def extract_contrasts(text, hooked_model, tokenizer, batch_size=1,max_length=256, max_new_tokens=100, cycle_number=0, device="cpu"):
    # ROUND 1 - for unexpected loops
    # for each sample
    natural_gens=[]
    natural_probs = []
    all_reps = []
    for i in tqdm(range(0, len(text), batch_size), desc="analysing samples", total=len(text)//batch_size):
        sample = text[i:i+batch_size]
        toked= tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        toked = {k: v.to(device) for k, v in toked.items()}
        o1=hooked_model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens)

        
        # are there repetitions?
        reps = [detect_cycles(tok[toked["input_ids"].shape[1]:].cpu(), return_index=True) for tok in o1]
        # if some repetitions are missing, we warn and print index
        if any(rep[0] is None for rep in reps):
            warn(f"Some repetitions were not detected in sample {i} (batch size {batch_size}). This may be due to the model not generating enough tokens or the repetition being too short.")
            for j, rep in enumerate(reps):
                if rep[0] is None:
                    print(f"Sample {i}, index {j} no repetition detected")

        # get model logits for the first repetition
        natural_prob = []
        # natural_logits = []
        natural_prob = [
            (logits := hooked_model(toked["input_ids"][j:j+1, :rep[3]+1]).logits).softmax(dim=-1)[:, -1, toked["input_ids"][j, rep[3]+1]].item()
            if rep[0] is not None else None
            for j, rep in enumerate(reps)
        ]
        # Batch decode all outputs at once for efficiency
        natural_gen=tokenizer.batch_decode(o1, skip_special_tokens=True)
        natural_gens.extend(natural_gen)
        natural_probs.extend(natural_prob)
        all_reps.extend(reps)

    # return text, o1, natural_probs, natural_logits, reps
    return text, natural_gens, natural_probs, None, all_reps
        

def main(
    model_name:str="EleutherAI/pythia-1.4b",
    base_path:str=None,
    revision:str=None,
    use_bfloat16:bool=False,
    seed:int=42,
    batch_size:int=1,
    max_length:int=256,
    max_new_tokens:int=256,
    ):
    # LOAD model using lens
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision) if not use_bfloat16 else \
        AutoModelForCausalLM.from_pretrained_no_processing(model_name, revision=revision, torch_dtype=torch.bfloat16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # probed_model.config.pad_token_id = tokenizer.pad_token_id

    # hooked_model = HookedTransformer.from_pretrained(
    #         model_name,
    #         hf_model=probed_model,
    #         tokenizer=tokenizer,
    #         n_embd=probed_model.config.hidden_size,
    #         n_layer=probed_model.config.num_hidden_layers,
    #         n_head=probed_model.config.num_attention_heads,
    #         vocab_size=probed_model.config.vocab_size,
    #         n_ctx=probed_model.config.max_position_embeddings,
    #         dtype=torch.bfloat16 if use_bfloat16 else None
    # )

    # hooked_model.eval()
    # del probed_model
    # hooked_model.set_use_attn_result(True)
    # torch.set_grad_enabled(False)

    # find all csv files in the base path
    
    # DATA --> part of the pile that we are not using for the repetition analyisis
    dataset = load_dataset("JeanKaddour/minipile")
    subset = dataset["train"].shuffle(seed=seed)
    # subset = subset.select(range(0, 100)) # 
    # subset = subset.select(range(0, len(subset) - 10000)) # all but last 10k datapoints
    subset = subset.select(range(len(subset) - 10000, len(subset))) # select last 10k datapoints

    if base_path is None:
        csv_files = [None]
        cycle_number = 0
    else:
        if not base_path.exists():
            raise ValueError(f"Base path {base_path} does not exist. Please provide a valid path.")
        csv_files = list(base_path.glob("*.csv"))

    for index_file in csv_files:
        if index_file is not None:
            # load the csv file
            indexes= pd.read_csv(index_file)["index"].tolist()

            # file name looks like this alluvial_origin_cycle0_t128_repeated_sentences.csv --> extract max_new_tokens from it
            match = re.search(r'_t(\d+)_', index_file.stem)
            if match:
                max_new_tokens = int(match.group(1))
            else:
                raise ValueError(f"Could not extract max_new_tokens from file name {index_file.stem}. Please provide a valid file name.")
            # extract cycle_number from the file name
            match = re.search(r'_cycle(\d+)_', index_file.stem)
            if match:
                cycle_number = int(match.group(1))
            else:
                raise ValueError(f"Could not extract cycle_number from file name {index_file.stem}. Please provide a valid file name.")
            # only keep the indexes that are in the subset
            subset = subset.select(indexes)["text"] # select only the indexes we are interested in
        else:
            indexes = list(range(len(subset)))  # if no file, use all indexes
            subset = subset["text"]  # use the whole dataset

        if len(subset) == 0:
            print(f"No data found for indexes {indexes} in file {index_file}. Skipping.")
            continue
        subset, natural_gens, natural_probs, _, reps = extract_contrasts(
            subset,
            model,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            cycle_number=cycle_number,
            device=device
        )

        # save the results in a csv file
        output_file = base_path / f"{index_file.stem}_{revision}.csv" if index_file is not None else base_path / f"all_indexes_{revision}.csv" if base_path is not None else f"results_{revision}.csv"
        results = {
            "text": subset,
            "natural_gen": natural_gens,
            "natural_probs": natural_probs,
            # "natural_logits": [logits.tolist() if logits is not None else None for logits in natural_logits],
            "cycle": [rep[0] for rep in reps],
            "cycle_size": [rep[1] for rep in reps],
            "cycle_count": [rep[2] for rep in reps],
            "cycle_start_index": [rep[3] for rep in reps],
        }
        for key in results:
            print(f"{key}: {len(results[key])} items")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")



    

if __name__ == "__main__":
    typer.run(main)
