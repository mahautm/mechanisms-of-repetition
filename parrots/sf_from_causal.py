import logging
from glob import glob
from pathlib import Path

import pandas as pd
import torch
import tqdm
import typer
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig

from parrots.archs import get_model
from parrots.cycle_detection import process_generation_column
from parrots.slot_filling import slot_fill
    
def main(
    data_path,
    model_name: str,
    output_path,
    batch_size: int=1,
    max_new_tokens: int=100,
    log_file: str="./logs",
    use_bnb: bool=False,
    use_accelerator: bool=False,
):
    dataset = load_from_disk(data_path)["sentence"]
    dataset = [d[0] for d in dataset]
    tr_dataloader = DataLoader(dataset, batch_size=batch_size)
    # load model
    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    model, tokenizer = get_model(
        model_name=model_name,
        quantization_config=bnb_config,
        device_map="balanced" if not use_accelerator else None,
    )
    if use_accelerator:
        accelerator = Accelerator()
        model, tokenizer, tr_dataloader = accelerator.prepare(model, tokenizer, tr_dataloader)
        device = accelerator.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ## END INITIALISATION
    
    ## RUN EVALUATION
    out = {"generated": []}
    for batch in tqdm.auto.tqdm(tr_dataloader):
        direct_follow, exact_match, nli_factual, generated = slot_fill(model, tokenizer, batch, None, max_new_tokens, device, None)
        # print(generated)
        out["generated"].extend(generated)
    out = pd.DataFrame(out)
    out["to_send"] = dataset
    out = process_generation_column(out, generation_col="generated", tokenizer=tokenizer)
    ## END EVALUATION
    ## SAVE --> include accelerator device if multi gpu, increment filename if already exists
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    if use_accelerator:
        filename = f"sf_causal_{accelerator.device.index}.csv"
    else:
        filename = "sf_causal.csv"
    pd.DataFrame(out).to_csv(output_path / filename)
    ## END SAVE

if __name__ == "__main__":
    typer.run(main)
