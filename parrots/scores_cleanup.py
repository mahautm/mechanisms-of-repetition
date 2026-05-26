import pandas as pd
import torch
from pathlib import Path
def main(data_file, score_file):
    # load clean data
    df=pd.read_csv(data_file)
    # load score data
    sf=pd.read_csv(score_file)
    scores=[]
    for i in range(len(df)):
        # add score to line
        value=df["to_send"][i]
        score=sf[sf["text"]==value]["confidence"]
        if len(score) == 1:
            scores.append(score.item())
        else:
            print(f"{len(score)} matching sentences found for {value} - no score will be given for this sentence")
            scores.append(None)
        # else:
        #     print(score)
        #     if all([i for i in score] == score.iloc[0]):
        #         print(f"Differences:{score}")
        #     else:
        #         print(f"different scores were found for the same sentence {score}") 

    df["probe_confidence"]=scores
    return df

if __name__ == "__main__":
    # .pt to .csv
    # scs = []
    # txts = []
    # for file in sorted(Path("/home/mmahaut/projects/parrots/outputs/_all_lama/").glob("hidden*.pt")):
    #     sc = torch.load(file)
    #     txt = torch.load(file.parent / f"None_uuids_{str(file)[-6:-3].split('_')[-1]}.pt")
    #     scs.extend(sc.cpu().to(torch.float32).detach().numpy())
    #     txts.extend(txt)
    # pd.DataFrame({"text": txts, "confidence": scs}).to_csv("/home/mmahaut/projects/parrots/outputs/_all_lama/scores.csv", index=False)
    # df = main("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_lama_sf/slot_filling_results.csv", "/home/mmahaut/projects/parrots/outputs/_all_lama/scores.csv")
    # df.to_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_lama_sf/scores_test.csv", index=False)

    scs = []
    txts = []
    for file in sorted(Path("/home/mmahaut/projects/parrots/outputs/_autoprompt/").glob("hidden*.pt")):
        sc = torch.load(file)
        txt = torch.load(file.parent / f"None_uuids_{str(file)[-7:-3].split('_')[-1]}.pt")
        scs.extend(sc.cpu().to(torch.float32).detach().numpy())
        txts.extend(txt)
    pd.DataFrame({"text": txts, "confidence": scs}).to_csv("/home/mmahaut/projects/parrots/outputs/_autoprompt/scores.csv", index=False)
    df = main("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results.csv", "/home/mmahaut/projects/parrots/outputs/_autoprompt/scores.csv")
    df.to_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/scores_test.csv", index=False)   
    # df=main("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results.csv","/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b/scores.csv")
    # df.to_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/scores_test.csv", index=False)
