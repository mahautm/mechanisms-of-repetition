# for a column "text" in a dataframe, for a given model, look at the distance between the text and every possible token in the vocabulary
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
def get_model(model_name, no_grad=True, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if no_grad:
        for p in model.parameters():
            p.requires_grad = False
    return model, tokenizer

def get_tokenizer(model_name, **kwargs):
    t = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t

def get_distance(model_name, df, batch_size=32, no_grad=True, **kwargs):
    model, tokenizer = get_model(model_name, no_grad=no_grad, **kwargs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    vocab = tokenizer.get_vocab()
    dists = np.zeros((len(df), len(vocab)))
    for i in tqdm(range(0, len(df), batch_size), position=0):
        batch = df.iloc[i:i+batch_size]
        texts = batch["to_send"].tolist()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_layer = outputs["hidden_states"]
            selected_token = inputs["input_ids"][:, -1]
            for token in tqdm(vocab, position=1, leave=False):
                token_id = vocab[token]
                token_embedding = model.get_input_embeddings().weight[token_id]
                # check size match
                distance = torch.norm(last_layer[-1][:,-1,:] - token_embedding, p=2, dim=1)
                dists[i:i+batch_size, token_id] = distance.cpu().numpy()
            # check selected token is the closest in the vocabulary
            # assert torch.argmax(torch.from_numpy(dists[j])).item() == selected_token[j].item(), f"Selected token {selected_token[j]} is not the closest in the vocabulary"
    return dists

if __name__ == "__main__":
    # How far away are the tokens from the last token in the sequence?
    # for autoprompts (both in domain and ood subjects) 
    # for natural language (both in domain and ood subjects)
    # plot all 4 distributions as boxplots in the same figure

    df_autop = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results.csv")
    df_autop_in = df_autop[df_autop["augmentation"]=="original"]
    df_autop_ood = df_autop[df_autop["augmentation"]=="unrelated"]

    df_nl = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results.csv")
    df_nl_in = df_nl[df_nl["augmentation"]=="original"]
    df_nl_ood = df_nl[df_nl["augmentation"]=="unrelated"]

    dists_autop_in = get_distance("facebook/opt-1.3b", df_autop_in)
    dists_autop_ood = get_distance("facebook/opt-1.3b", df_autop_ood)
    dists_nl_in = get_distance("facebook/opt-1.3b", df_nl_in)
    dists_nl_ood = get_distance("facebook/opt-1.3b", df_nl_ood)

    all_dists = np.concatenate([dists_autop_in, dists_autop_ood, dists_nl_in, dists_nl_ood], axis=0)
    all_dists = pd.DataFrame(all_dists)
    all_dists["augmentation"] = ["autoprompts_in"]*len(dists_autop_in) + ["autoprompts_ood"]*len(dists_autop_ood) + ["natural_language_in"]*len(dists_nl_in) + ["natural_language_ood"]*len(dists_nl_ood)
    # plot all 4 distributions as boxplots in the same figure
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(data=all_dists.melt(id_vars=["augmentation"]), x="augmentation", y="value", hue="augmentation")
    ax.set_xlabel("Augmentation")
    ax.set_ylabel("Distance")
    plt.savefig("distance_increase3.png")


                