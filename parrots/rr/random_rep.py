from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import torch
import warnings
from tqdm import tqdm
import typer
from transformers import BitsAndBytesConfig
import time
from threading import Lock

def data_generation(tokenizer, cycle_size, n_cycles, dataset_size, batch_size, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer_max_value = tokenizer.vocab_size

    if dataset_size % batch_size != 0:
        warnings.warn(f"Dataset size {dataset_size} is not divisible by batch size {batch_size}. Truncating dataset size.")

    def token_generator():
        for _ in range(dataset_size//batch_size):
            seq = torch.randint(0, tokenizer_max_value, (batch_size, cycle_size))
            # extend seq by repeating it n_cycles times
            seq = seq.repeat((1, n_cycles))
            yield seq

    return token_generator()

def main(
    cycle_size: int,
    n_cycles: int,
    dataset_size: int,
    batch_size: int,
    model_name: str,
    revision:str=None,
    cache_dir:str=None,
    use_bnb: bool=False,
    deactivate_tqdm: bool=False,
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Configure bits and bytes if required
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        revision=revision, 
        cache_dir=cache_dir, 
        load_in_4bit=use_bnb,
    )
    if not use_bnb:
        model = model.to(device)
    
    tokenizer_lock = Lock()

    def load_tokenizer(model_name, revision, cache_dir):
        with tokenizer_lock:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        return tokenizer

    tokenizer = load_tokenizer(model_name, revision, cache_dir)


    # eos and pad
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    _gen = data_generation(tokenizer, cycle_size, n_cycles, dataset_size, batch_size)

    score = 0
    log = []
    for batch in tqdm(_gen, total=dataset_size//batch_size, desc="Processing batches", disable=deactivate_tqdm):
        batch = batch.to(device)
        o = model.generate(batch, max_length=100)
        # check if model generation is just repeating the input
        sequence = batch[:, :cycle_size]
        success = (o[:, :sequence.shape[1] * (o.shape[1]//cycle_size)] == sequence.repeat((1, o.shape[1]//cycle_size))).all(dim=1)
        if success.sum() < batch_size:
            seqs = sequence[success == 0, :].cpu().tolist()
            os = o[success == 0, :].cpu().tolist()
            log.extend(list(zip(seqs, os)))
        score += success.sum().item()
           

    print(f"random_rep params are: n_cycles={n_cycles}, cycle_size={cycle_size}, dataset_size={dataset_size}, batch_size={batch_size}, model_name={model_name}, revision={revision}, cache_dir={cache_dir}")
    print(f"Score: {score}/{dataset_size}")
    # print(f"Failed sequences: {log}")
    # print decoded failed sequences
    for seq, o in log:
        print("Sequence:", tokenizer.decode(seq))
        print("Tok sequence:", seq)
        print("Output:", tokenizer.decode(o))
        print("Tok output:", o)


if __name__ == "__main__":
    typer.run(main)