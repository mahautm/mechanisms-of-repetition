import torch
import typer
from typing import Optional
from tuned_lens.nn.lenses import TunedLens
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.plotting import PredictionTrajectory
from paramem.data import prepare_data, load_csv_data, load_pile_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import pickle


def plot_ranks(df, save_path="my_lenses/ranks.png", label=None):
    n_layers = len(df.iloc[0]["ranks"])
    # df["ranks"] = df["ranks"].apply(lambda x: [sum(x[i])/len(x[i]) for i in range(n_layers)])
    df["ranks"] = df["ranks"].apply(lambda x: [y[-1] for y in x])
    df[[f"layer_{i}" for i in range(n_layers)]] = pd.DataFrame(df["ranks"].tolist(), index= df.index)
    df = pd.melt(df, id_vars=["ranks"], value_vars=[f"layer_{i}" for i in range(n_layers)], var_name="Layer", value_name="Rank")
    # make columns into categorical variables for seaborn
    sns.lineplot(data=df, x="Layer", y="Rank", label=label)
    # angle x labels
    plt.xticks(rotation=45)
    plt.title("Ranks")
    # axis
    plt.xlabel("Layer")
    plt.ylabel("Rank")
    plt.savefig(save_path)

def main(
    model_name:str,
    lens_name:str,
    batch_size:int,
    input_key:str,
    following_key:str,
    dataset_path:str,
    outpath:str,
    device:Optional[str] = "cpu",
    dry_run:bool = False,
    plot_save_path:Optional[str] = None,
    use_kn_threshold:bool = True,
    repetition:Optional[float] = None,
    ):
    device = torch.device(device)
    # To try a diffrent modle / lens check if the lens is avalible then modify this code
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=lens_name, map_location=device)
    tuned_lens = tuned_lens.to(device)

    # dataset_path = "/home/mmahaut/projects/paramem/data/wikidata_Met7.csv"
    # input_key = "query"
    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
        if repetition is not None:
            df=df[df["entire_subject_in_generated"]==repetition]
        # df=df[df["augmentation"]=="unrelated"]
    else:
        raise ValueError("data_file should be a csv file")
    if dry_run:
        if len(df) > 100:
            df = df.sample(100, random_state=42)
        else:
            print(f"Tried to sample 100 but only {len(df)} rows available. Continuing with {len(df)} rows.")
    ranks = []
    for i in tqdm.tqdm(range(len(df))):
        tok_input = tokenizer.encode(df.iloc[i][input_key] + df.iloc[i][following_key], return_tensors="pt")
        _o = model(tok_input)
        # get five most likely tokens from the model
        ans_len = len(tokenizer(df.iloc[i][following_key])["input_ids"])
        
        toked_targets = torch.cat([tok_input, torch.tensor([[tokenizer.eos_token_id]], device=tok_input.device)], dim=1).squeeze()
        toked_targets = toked_targets[1:]

        pred = PredictionTrajectory.from_lens_and_model(
            tuned_lens,
            model,
            tokenizer=tokenizer,
            input_ids=tok_input.squeeze(),
            targets=toked_targets,
        ).slice_sequence([-ans_len,-ans_len+1])
        ranks.append(pred.rank().stats)
    df["ranks"] = ranks
    print(df.head(1)["ranks"])
    # save ranks
    with open(outpath, 'wb') as f:
        pickle.dump(df, f)
    if plot_save_path is not None:
        plot_ranks(df, plot_save_path)#, label=f"kn_threshold={kn_threshold}")

if __name__ == "__main__":
    # typer.run(main)
    main(
        model_name="facebook/opt-1.3b",
        lens_name="/home/mmahaut/projects/parrots/my_lenses/opt1",
        batch_size=1,
        input_key="to_send",
        following_key="subject",
        dataset_path="./outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results.csv",
        outpath="./outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/norep_ranks.pkl",
        device="cpu",
        dry_run=True,
        plot_save_path="./outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/ranks.png",
        repetition=True,
    )
    # main(
    #     model_name="facebook/opt-1.3b",
    #     lens_name="/home/mmahaut/projects/parrots/my_lenses/opt1",
    #     batch_size=1,
    #     input_key="to_send",
    #     following_key="subject",
    #     dataset_path="./outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results.csv",
    #     outpath="./outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/rep_ranks.pkl",
    #     device="cpu",
    #     dry_run=True,
    #     plot_save_path="./outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/ranks.png",
    #     repetition=False,
    # )
    # with open("my_lenses/ranks.pkl", 'rb') as f:
    #     df = pickle.load(f)
    # plot_ranks(df)