import pandas as pd
from tuned_lens.nn.lenses import TunedLens
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
    n_words=[]
    if min_pad_size is not None:
        max_len=min_pad_size
    else:
        max_len=0
    for i, text in df["combined"].items():
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
        max_len=max(max_len, len(pred.log_probs[-1,:,:]))
        n_words.append(len(pred.log_probs[-1,:,:]))
        trajectories.extend(pred.log_probs[-1,:,:].reshape(-1, 50272))
    return trajectories, n_words


if __name__ == "__main__":
    model_name="facebook/opt-1.3b"
    lens_name="/home/mmahaut/projects/parrots/my_lenses/opt1"
    df = pd.read_csv("./outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results_with_cycles.csv")
    # only keep the rows with cycles
    df_a = df[df["cycle_count"] > 1].sample(n=50)
    # make the space out of a random sample of the generations
    df_b = df.sample(n=50)
    df = pd.concat([df_a, df_b])
    df["combined"]=df["to_send"]+" "+df["generated"]


    device="cuda"
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tok_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=lens_name, map_location=device)
    tok_lens = tok_lens.to(device)

    trajectories, n_words_in_traj = layer_trajectories(df,model,tok_lens,tokenizer)

    embedding = MDS(n_components=2, normalized_stress='auto')
    X_transformed = embedding.fit_transform(np.array(trajectories).reshape(-1,50272).astype(np.float64))
    df2=pd.DataFrame()
    df2["X"]=X_transformed[:,0]
    df2["Y"]=X_transformed[:,1]
    _split_idx=len(n_words_in_traj)//2
    df2["isolated"]=["isolated"]*sum(n_words_in_traj[_split_idx:])+["dupplicates"]*sum(n_words_in_traj[:_split_idx])

    k=4
    for i in range(k):
        sns.kdeplot(data=df2, x="X", y="Y", fill=True, common_norm=False, alpha=0.5, palette="crest", hue="isolated")
        _summed_words=sum(n_words_in_traj[:i])
        # plt.text(df2["X"].iloc[_summed_words+j], df2["Y"].iloc[_summed_words+j], str(j), fontsize=12, color="black" if i==0 else "grey")
        # make a palette with the same number of colors as the number of cycles
        palette=sns.color_palette("Spectral", n_colors=len(n_words_in_traj))
        # change color of each cycle
        tok_texts = tokenizer(df["combined"].iloc[i], return_tensors="pt", padding=True, truncation=True)
        tok_cycle = tokenizer(df["cycle"].iloc[i], return_tensors="pt", padding=True, truncation=True)
        # every time we see a cycle, we change the color
        cycle=[0]*len(tok_texts["input_ids"])
        cyc_number=1
        for j in range(len(tok_texts["input_ids"])-len(tok_cycle["input_ids"])):
            if tok_texts["input_ids"][j:j+len(tok_cycle["input_ids"])]==tok_cycle["input_ids"]:
                cycle[j:j+len(tok_cycle["input_ids"])]=[cyc_number]*len(tok_cycle["input_ids"])
                cyc_number+=1
        cyc_palette=[sns.color_palette("Spectral", n_colors=len(palette))[c % len(palette)] for c in cycle]
        # map numbers in cyc_palette to valid color names or RGB tuples
        # cols = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "grey", "black"]
        # cyc_palette = [cols[c] for c in cycle]

        plt.plot(df2["X"].iloc[_summed_words:_summed_words+n_words_in_traj[i]], df2["Y"].iloc[_summed_words:_summed_words+n_words_in_traj[i]], palette=cyc_palette)
        plt.suptitle(df["cycle"].iloc[i], fontsize=8)
        plt.savefig(f"/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/LOOP{i}.png")
        plt.clf()
