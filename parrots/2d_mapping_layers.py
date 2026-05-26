import pandas as pd
from tuned_lens.plotting import PredictionTrajectory
from sklearn.manifold import MDS
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens
from tuned_lens.plotting import PredictionTrajectory
import seaborn as sns
from matplotlib import pyplot as plt

def extract_logits(df,model,tokenizer):
    trajectories=[]
    max_len=0
    for _, text in df["to_send"].items():
        toked=tokenizer(text)
        trajectory=[]
        for i in range(len(toked["input_ids"])):
            _t = {}
            for k,v in toked.items():
                _t[k]=torch.tensor([v[:i+1]]).to(device)
            o=model(**_t)
            trajectory.append(o.logits[0,-1,:].cpu().detach().numpy())
        trajectories.append(trajectory)
        if len(trajectory) > max_len:
            max_len=len(trajectory)
    if min_pad_size is not None:
        max_len=max(min_pad_size, max_len)
    padded_t=[]
    for t in trajectories:
        _t=np.pad(t, ((0,max_len-len(t)),(0,0)), mode="edge")
        padded_t.append(_t)
    trajectories=np.array(padded_t)
    print(trajectories.shape)
    return trajectories

def plot_trajectories(trajectories):
    embedding = MDS(n_components=2, normalized_stress='auto')
    X_transformed = embedding.fit_transform(trajectories)
    df2=pd.DataFrame()
    df2["X"]=X_transformed[:,0]
    df2["Y"]=X_transformed[:,1]
    _hues=[]
    for i in range(len(df2)//trajectories.shape[1]):
        _hues.extend([i]*trajectories.shape[1])
    df2["hue"]=_hues
    df2["style"]=[0]*n_samples*trajectories.shape[1]+[1]*n_samples*trajectories.shape[1]
    sns.lineplot(df2, x="X",y="Y",hue="hue",style="style", palette=sns.color_palette("Spectral", as_cmap=True))
    plt.savefig(f"/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/TEST.png")
    plt.clf()

def layer_trajectories(df,model,tok_lens, tokenizer,min_pad_size=None):
    trajectories=[]
    for _, text in df["to_send"].items():
        toked=tokenizer(text)
        _t = {k: torch.tensor([v]).to(device) for k, v in toked.items()}
        o = model(**_t)
        pred = PredictionTrajectory.from_lens_and_model(
            tok_lens,
            model,
            tokenizer=tokenizer,
            input_ids=toked["input_ids"],
            targets=toked["input_ids"][1:] + [tokenizer.eos_token_id],
        )
        trajectories.append(pred.log_probs[:,-1,:])
    return trajectories


if __name__ == "__main__":
    model_name="facebook/opt-1.3b"
    lens_name="/home/mmahaut/projects/parrots/my_lenses/opt1"
    df = pd.read_csv("./outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results.csv")
    device="cuda"
    n_samples=200
    # sample 100 non-repeated datapoints
    nrep=df[df["partial_subject_in_generated"]==False].sample(n_samples)
    # sample 100 repeated datapoints
    rep=df[df["partial_subject_in_generated"]==True].sample(n_samples)
    # present path taken by representation after each token
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tok_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=lens_name, map_location=device)
    tok_lens = tok_lens.to(device)

    trajectories=np.array(layer_trajectories(rep,model,tok_lens,tokenizer))
    ntrajectories=np.array(layer_trajectories(nrep,model,tok_lens,tokenizer))
    all_tr=np.concatenate([trajectories, ntrajectories])
    fig, axes = plt.subplots(trajectories.shape[1], 1, figsize=(10, trajectories.shape[1] * 5))
    
    for i in range(trajectories.shape[1]):
        embedding = MDS(n_components=2, normalized_stress='auto')
        X_transformed = embedding.fit_transform(all_tr[:, i, :])
        df2 = pd.DataFrame()
        df2["X"] = X_transformed[:, 0]
        df2["Y"] = X_transformed[:, 1]
        _hues = ["repeating"] * (len(df2) // 2) + ["non-repeating"] * (len(df2) // 2)

        examples = [[j] * 2 for j in range(n_samples)]
        examples = [item for sublist in examples for item in sublist]
        df2["example"] = examples
        df2["type"] = _hues

        sns.kdeplot(data=df2, x="X", y="Y", hue="type", fill=True, common_norm=False, alpha=0.5, palette="crest", ax=axes[i])
        axes[i].set_title(f'Layer {i}')
    
    plt.tight_layout()



    plt.savefig(f"/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/layer_per_layer_mds.png")
    plt.clf()
    # plot using (MDS and PCA)