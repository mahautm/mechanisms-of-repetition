from transformers import AutoModelForCausalLM, AutoTokenizer
from parrots.cycle_detection import detect_cycles
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
import typer
from pathlib import Path

def first_pass(model, tokenizer, subset, max_new_tokens, batch_size, device):
    # first pass --> generate continuation to pile text, detect cycles
    output = {}
    output["generated"] = []
    output["cycle"] = []
    output["cycle_size"] = []
    output["cycle_count"] = []
    output["toked_input"] = []
    output["toked_transition"] = []


    for batch_idx in tqdm(range(0, len(subset), batch_size), desc="First pass"):
        batch = subset[batch_idx:batch_idx+batch_size]
        inputs = batch["text"]
        toked = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**toked, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        detok = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # cycle, cycle_size, cycle_count = [], [], []
        for line in range(outputs.shape[0]):
            # is the input repeated in the generated text?
            if (outputs[line,:toked["input_ids"].shape[1]] == toked["input_ids"][line,...]).all():
                o=outputs[line,toked["input_ids"].shape[1]:]
            else:
                o=output[line,:]
            c, cs, cc, csi = detect_cycles(o, return_index=True)
            csi = csi if csi != -1 and csi != 0 else None
            output["generated"].append(detok[line])
            output["cycle"].append(c.cpu().numpy().tolist() if c is not None else None)
            output["toked_input"].append(toked["input_ids"][line,...].cpu().numpy().tolist())
            output["toked_transition"].append(o[:csi].cpu().numpy().tolist() if csi is not None else [])
            output["cycle_size"].append(cs)
            output["cycle_count"].append(cc)
        del toked, outputs, detok
        torch.cuda.empty_cache()

    return output

def second_pass(model, tokenizer, subset, cycle_size, max_new_tokens, inputs, n_perturbations, batch_size, device):
    # second pass --> for cycles, generate continuation to pile text + cycle with token of different probability
    output = {}
    output["entropy"] = []
    output["perturbator_prob"] = []
    output["rank"] = []
    output[f"perturbator_token"] = []
    output[f"perturbated_output"] = []
    output["prev_cycle"] = []
    print("len inputs", len(inputs[list(inputs.keys())[0]]))
    for batch_idx in tqdm(range(0, len(inputs[list(inputs.keys())[0]]), batch_size), desc="Second pass"):
        # text_input = subset[batch_idx:batch_idx+batch_size]["text"]
        cycle = inputs["cycle"][batch_idx:batch_idx+batch_size]
        def condition(x):
            return x is not None and x != [] and pd.notna([x]).all()
        idxs = [i for i, x in enumerate(cycle) if condition(x)]
        cycle = [cycle[i] for i in idxs]

        # token, output, exact prob, entropy

        # TODO: this only works for batch 1, need to fix for batch > 1
        toked = torch.tensor(inputs["toked_input"][batch_idx:batch_idx+batch_size]).to(device)
        trans = torch.tensor(inputs["toked_transition"][batch_idx:batch_idx+batch_size]).to(device)

        with torch.no_grad():
            logits = model(toked).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1).item()

            # get the most likely token below p in perturbation_probs
            for p_idx in range(n_perturbations):
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # cumulative_probs = torch.cumsum(sorted_probs.double(), dim=-1).float()
                # token = sorted_indices[cumulative_probs <= p]
                # token = token[0] if token.shape[0] > 0 else sorted_indices[0][1]
                token = sorted_indices[0][p_idx]
                token = token.unsqueeze(0).unsqueeze(0)

                # chosen_prob = sorted_probs[cumulative_probs <= p][0] if sorted_indices[cumulative_probs <= p].shape[0] > 0 else sorted_probs[0][1]
                chosen_prob = sorted_probs[0][p_idx]
                chosen_prob = chosen_prob.unsqueeze(0).unsqueeze(0)
                cycles = [c*cycle_size for c in cycle if c is not None]
                c = torch.tensor(cycles).to(device)

                perturbed_input = torch.cat([toked, trans, c, token], dim=-1)
                perturbed_input = {"input_ids": perturbed_input.to(torch.long), "attention_mask": torch.ones_like(perturbed_input).to(torch.long)}
                outputs = model.generate(**perturbed_input, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
                # remove the input
                outputs = outputs[:,perturbed_input["input_ids"].shape[1]:]

                # save results
                output["entropy"].append(entropy)
                output["perturbator_prob"].append(chosen_prob.cpu().numpy().tolist())
                output["rank"].append(p_idx)
                output[f"perturbator_token"].append(token.cpu().numpy().tolist())
                output[f"perturbated_output"].append(outputs.squeeze(0).cpu().numpy().tolist())
                output["prev_cycle"].append(cycles)


    return output

def main(
    rank_number:int=0, 
    n_ranks:int=1, 
    model_name:str="EleutherAI/pythia-1.4B", 
    batch_size:int=1, 
    max_new_tokens:int=500, 
    cycle_size:int=2, 
    skip_phase_1:bool=False,
    skip_phase_2:bool=False,
    save_path:str=".",
    load_in_8bit:bool=False,
    seed:int=42,
    ):
    # perturbation_probs = [0.95, 0.99, 0.999, 0.9999, 0.99999]
    n_perturbations=100
    Path(save_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=load_in_8bit).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("JeanKaddour/minipile")
    # TODO: add assert that we're not touching the last 10k datapoint which are used for lens training
    subset = dataset["train"].shuffle(seed=seed)
    subset_size = len(subset)//n_ranks
    subset = subset.select(range(subset_size * rank_number, subset_size * (rank_number + 1)))
    # only keep 50 first words in every text
    subset = subset.map(lambda x: {"text": x["text"][:50]})
    print(f"Rank {rank_number} has {len(subset)} samples")

    # first pass
    if skip_phase_1:
        df = pd.read_csv(Path(save_path) / f"cycle_{cycle_size}_results_{rank_number}.csv")
        df["cycle"] = df["cycle"].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
        df["toked_input"] = df["toked_input"].apply(lambda x: pd.eval(x) if isinstance(x, str) else [])
        df["toked_transition"] = df["toked_transition"].apply(lambda x: pd.eval(x) if isinstance(x, str) else [])
        output = df.to_dict(orient="list")
    else:
        output = first_pass(model, tokenizer, subset, max_new_tokens, batch_size, device)
        # intermediary save
        df = pd.DataFrame(output)
        df.to_csv(Path(save_path) / f"cycle_{cycle_size}_results_{rank_number}.csv", index=False)
    
    
    print("Proportion of cycles detected in first pass:", sum([ x > 0 for x in output["cycle_size"]])/len(output["cycle"]))
    if skip_phase_2:
        return
    # second pass
    output2 = second_pass(model, tokenizer, subset, cycle_size, max_new_tokens, output, n_perturbations, batch_size, device)
    df2 = pd.DataFrame(output2)
    _res = df2["perturbated_output"].apply(lambda x: detect_cycles(torch.tensor(x)) if x is not None else (None, None, None))
    df2[f"cycle"], df2[f"cycle_size"], df2[f"cycle_count"] = zip(*_res)
    df2["generation"] = df2["perturbated_output"].apply(lambda x: tokenizer.batch_decode([x]) if x is not None else None)
    df2 = df2.drop(columns=["perturbated_output"])
    print("Proportion of cycles detected in second pass:", sum([ x > 0 for x in df2["cycle_size"]])/len(df2["cycle"]))
    # see proportion of cycles which are the same as in the first pass
    # df = pd.DataFrame(np.repeat(df.values, n_perturbations, axis=0), columns=df.columns)
    print(len(df), len(df2))
    # matching = df["cycle"] == df2["cycle"]
    # print("Proportion of cycles detected in second pass that are the same as in the first pass:", sum(matching)/len(matching))

    
    # save final results
    df2.to_csv(Path(save_path) /f"cycle_{cycle_size}_perturbation_results_{rank_number}.csv", index=False)
                





if __name__ == "__main__":
    typer.run(main)
