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
from parrots.nli import NLI


def slot_fill(model, tokenizer, input_sentence, expected_output, max_new_tokens, device, nli=None, top_p=None, use_icl=False):
    """
    get a batch of sentences from sentence with missing completion
    generate an amount of words greedily that corresponds to the possible answer
    check and label if answer is correct

    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : transformers.PreTrainedTokenizerFast
    input_sentence : list of str
    expected_output : list of str
    max_new_tokens : int
    device : torch.device  
    

    Returns
    -------
    direct_follow : list of bool
    exact_match : list of bool
    nli_factual : list of bool
    generated : list of str
    """
    if use_icl:
        input_sentence = [(text + "\n") * 4 + text for text in input_sentence]
    # prepare model input and run model
    input_tok = tokenizer(
        input_sentence, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if top_p is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = top_p
    
    gen_o = model.generate(**input_tok, **gen_kwargs)
    generated = tokenizer.batch_decode(gen_o, skip_special_tokens=True)

    # Testing the query is not in the generated text, some models do that
    if input_sentence[0] in generated[0]:
        for i in range(len(input_sentence)):
            if input_sentence[i] in generated[i]:
                generated[i] = generated[i].replace(input_sentence[i], "")

    if expected_output is not None:
        assert len(input_sentence) == len(expected_output), "input and expected output must have the same length"

        # direct follow, if the expected output is the very next word
        direct_follow = []
        for i in range(len(input_sentence)):
            _exp = " " + expected_output[i] # !! FIXME hardcoded common missing space
            _len = min(len(generated[i]), len(_exp))
            direct_follow.append(_exp[:_len] in generated[i][:_len])

        # exact matching, if the expected output is in the generated text
        exact_match = [expected_output[i] in _gen for i, _gen in enumerate(generated)]

        # nli matching (takes into account synonims)
        _truth = [
            f"{input_sentence[i]} {expected_output[i]}" for i in range(len(input_sentence))
        ]
        # nli = NLI()
        if nli is None:
            nli = NLI()
        nli_factual = nli.check_equivalence(_truth, generated)

        # logging
        logging.debug(f"input: {input_sentence}")
        logging.debug(f"generated: {generated}")
        logging.debug(f"expected: {expected_output}")
        logging.debug(f"direct_follow: {direct_follow}")
        logging.debug(f"exact_match: {exact_match}")
        logging.debug(f"nli_factual: {nli_factual}")
    else:
        direct_follow = [None for _ in range(len(input_sentence))]
        exact_match = [None for _ in range(len(input_sentence))]
        nli_factual = [None for _ in range(len(input_sentence))]

    return direct_follow, exact_match, nli_factual, generated
    
def main(
    data_path,
    model_name: str,
    output_path,
    batch_size: int=1,
    max_new_tokens: int=100,
    log_file: str="./logs",
    use_bnb: bool=False,
    use_accelerator: bool=False,
    top_p: float=None,
    use_icl: bool=False,
):
    ## INITIALISATION
    # initialise logging
    Path(log_file).parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file,
    )

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
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_accelerator:
        accelerator = Accelerator()
        model, tokenizer, tr_dataloader = accelerator.prepare(model, tokenizer, tr_dataloader)
        device = accelerator.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ## END INITIALISATION
    
    ## RUN EVALUATION
    out = {key: [] for key in ["generated", "direct_follow", "exact_match", "nli_factual"]}
    nli = NLI()
    for batch in tqdm.auto.tqdm(tr_dataloader):
        direct_follow, exact_match, nli_factual, generated = slot_fill(model, tokenizer, batch["to_send"], batch["corr_answer"], max_new_tokens, device, nli, top_p=top_p, use_icl=use_icl)
        out["generated"].extend(generated)
        out["direct_follow"].extend(direct_follow)
        out["exact_match"].extend(exact_match)
        out["nli_factual"].extend(nli_factual)
    out = pd.DataFrame(out)
    out = pd.concat([df, out], axis=1)
    ## END EVALUATION
    ## SAVE --> include accelerator device if multi gpu, increment filename if already exists
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    if use_accelerator:
        filename = f"slot_filling_results_{accelerator.device.index}.csv"
    else:
        filename = "slot_filling_results.csv"
    # i = 0
    # while (output_path / filename).exists():
    #     logging.warning(f"file {output_path / filename} already exists, incrementing filename")
    #     i += 1
    #     filename = f"slot_filling_results_{accelerator.device.index}_{i}.csv"

    pd.DataFrame(out).to_csv(output_path / filename)
    ## END SAVE

if __name__ == "__main__":
    typer.run(main)
