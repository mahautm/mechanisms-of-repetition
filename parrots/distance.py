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

def plot_distance_increase(dists, output_file, hue=None):
# from a matrix recording distances shaped [sequences, tokens], plot distance increase
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))
    # reorder each line to have increasing distance
    dists = np.sort(dists, axis=1)
    # plot average distance increase for each token in the vocabulary, averaged over all sequences, with a 95% confidence interval
    # the x-axis is the token index in the vocabulary, the y-axis is the average distance increase
    sns.lineplot(x=range(dists.shape[1]), y=np.mean(dists, axis=0), hue=hue)
    # dashes=False, ax=ax, errorbar=('ci', 95))
    ax.set_xlabel("Token")
    ax.set_ylabel("Average distance increase")
    plt.savefig(output_file)

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
    df = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results.csv")
    df1 = df[df["partial_subject_in_generated"]==True]
    dists = get_distance("facebook/opt-1.3b", df1)
    plot_distance_increase(dists, "distance_increase.png")
    df2 = df[df["partial_subject_in_generated"]==False]
    dists2 = get_distance("facebook/opt-1.3b", df2)
    all_dists = np.concatenate([dists, dists2], axis=0)
    all_dists = pd.DataFrame(all_dists)
    all_dists["partial_subject_in_generated"] = ["True"]*len(dists) + ["False"]*len(dists2)
    plot_distance_increase(dists, "distance_increase2.png", hue="partial_subject_in_generated") 

                