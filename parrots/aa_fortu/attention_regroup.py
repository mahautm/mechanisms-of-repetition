import glob
import json
# entropy
import numpy as np
from scipy.stats import entropy

def main():
    file_path="/home/mmahaut/projects/exps/parr/EleutherAI/pythia-1.4B/rraa2"
    files = glob.glob(file_path + "/*step100000_att_cycles.json")
    # get index number
    # print a graph of variation accross indexes
    # save average for each index
    # regroup the files
    grouped_attentions = {}
    for file in files:
        with open(file, "r") as f:
            attentions = json.load(f)
        for k, v in attentions.items():
            if k != "params":
                if k not in grouped_attentions:
                    grouped_attentions[k] = []
                grouped_attentions[k].append(v)

    total_acts = sum([sum(v) for k, v in grouped_attentions.items()])
    print(total_acts)
    normed_acts = {k: sum(v)/total_acts for k, v in grouped_attentions.items()}
    # 25 biggest heads
    selected_heads = [k for k, v in sorted(normed_acts.items(), key=lambda x: x[1], reverse=True)[:25]]
    print(selected_heads)

    # save the selected heads
    with open(file_path + "/average_selected_heads.json", "w") as f:
        json.dump(normed_acts, f)

    # select lowest entropy
    ent_acts = {k: entropy(v) for k, v in grouped_attentions.items()}
    e_selected_heads = [k for k, v in sorted(ent_acts.items(), key=lambda x: x[1], reverse=True)[:25]]
    print(e_selected_heads)


if __name__ == "__main__":
    main()