import logging
from glob import glob
from pathlib import Path

import pandas as pd
import torch
import tqdm
import typer
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig

from parrots.archs import get_model
from cycle_detection import process_generation_column
from slot_filling import slot_fill
    
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
    ## INITIALISATION
    # initialise logging
    # Path(log_file).parent.mkdir(exist_ok=True, parents=True)
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     filename=log_file,
    # )

    # load data
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except:
        df = pd.read_csv(data_path, encoding="latin-1")
    # fill missing values
    else:
        # find encoding
        logging.warning("data not loaded, check encoding")
    # cast Nonetype to str
    df["subj_position"] = df["subj_position"].fillna(-1)
    df["subj_position"] = df["subj_position"].astype(int)
    df["num_tokens_obj"] = df["num_tokens_obj"].fillna(-1)
    df["num_tokens_obj"] = df["num_tokens_obj"].astype(int)
    df["num_tokens_sub"] = df["num_tokens_sub"].fillna(-1)
    df["num_tokens_sub"] = df["num_tokens_sub"].astype(int)
    df["unrelated_origin"] = df["unrelated_origin"].fillna(-1)
    df["unrelated_origin"] = df["unrelated_origin"].astype(str)
    df = df.fillna("")
    
    df.rename(columns={"generated": "generated_old", "cycle": "cycle_old", "cycle_count": "cycle_count_old", "cycle_size": "cycle_size_old"}, inplace=True)
    df["transition"] = df.apply(lambda x: x['generated_old'].split(x["cycle_old"])[0] if x["cycle_old"] != "" else "", axis=1)
    df = df[df["transition"] != ""]

    dataset = Dataset.from_pandas(df)
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
        direct_follow, exact_match, nli_factual, generated = slot_fill(model, tokenizer, batch["transition"], None, max_new_tokens, device, None)
        print(generated)
        out["generated"].extend(generated)
    out = pd.DataFrame(out)
    out = pd.concat([df, out], axis=1)
    out = process_generation_column(out, generation_col="generated", tokenizer=tokenizer)
    ## END EVALUATION
    ## SAVE --> include accelerator device if multi gpu, increment filename if already exists
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    if use_accelerator:
        filename = f"sf_transitions_{accelerator.device.index}.csv"
    else:
        filename = "sf_transitions.csv"
    pd.DataFrame(out).to_csv(output_path / filename)
    ## END SAVE

if __name__ == "__main__":
    typer.run(main)
