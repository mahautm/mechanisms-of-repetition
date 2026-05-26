from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import typer
from plotly.colors import sample_colorscale
import plotly.graph_objects as go


def plot_from_proba(base_path, heatmap_type="natural", cycle_size=0, t_size=0):
    # collect logs in base_path
    logs = []
    for f in Path(base_path).glob(f"*_{cycle_size}_*_{t_size}.out"):
        with open(f, "r") as file:
            logs.append(file.read())
    # Extract heatmap elements from logs
    heatmaps = []
    for log in logs:
        # layer 23 natural heatmap: 
        matches = re.findall(fr"layer (\d+) {heatmap_type} heatmap: (.+)", log)
        for layer, match in matches:
            if match == "None":
                heatmap = None
            else:
                heatmap = np.array(eval(match))
            heatmaps.append((int(layer), heatmap))
    heatmaps.sort(key=lambda x: x[0])
    heatmaps = [heatmap for _, heatmap in heatmaps]
    return heatmaps

def plot_from_logs(base_path, cycle_size=0, t_size=0):
    # collect logs in base_path
    logs = []
    for f in Path(base_path).glob(f"*_{cycle_size}_*_{t_size}.out"):
        with open(f, "r") as file:
            logs.append(file.read())
    # Extract heatmap elements from logs
    heatmaps = []
    for log in logs:
        # layer 23 natural heatmap: 
        matches = re.findall(fr"layer (\d+) data index: (.+)", log)
        for layer, match in matches:
            if match == "None":
                heatmap = None
            else:
                heatmap = np.array(eval(match))
            heatmaps.append((int(layer), heatmap))
    heatmaps.sort(key=lambda x: x[0])
    heatmaps = [heatmap for _, heatmap in heatmaps]
    return heatmaps

def decentralised_to_csv():
    path = "/home/mmahaut/projects/parrots/outputs/EleutherAI/pythia-1.4b/step10000/"
    model_name = "EleutherAI/pythia-1.4b"
    steps = [1, 1000, 5000, 7000, 10000, 100000]
    steps = [f"step{i}" for i in steps]
    cycle_count = list(range(0, 5))
    t_size=[16,32,64]
    # plot from logs
    csv={}
    csv["cycle_size"] = []
    csv["t_size"] = []
    csv["step"] = []
    csv["data_index"] = []
    for i in range(0, 16):
        for l in range(0, 24):
            csv[f"nat_prob_l{l}_h{i}"] = []
            csv[f"icl_prob_l{l}_h{i}"] = []
    for cycle_size in cycle_count:
        for t in t_size:
            for step in steps:
                path = f"/home/mmahaut/projects/parrots/outputs/{model_name}/{step}/"
                data = plot_from_logs(path, cycle_size=cycle_size, t_size=t)
                data = data[0] if len(data) > 0 else None
                # format list of indexes for csv
                data = [int(i) for i in data]
                hm_icl = plot_from_proba(path, cycle_size=cycle_size, t_size=t, heatmap_type="icl")
                # print(hm_icl)
                # input("Press enter to continue...")
                hm_nat = plot_from_proba(path, cycle_size=cycle_size, t_size=t)
                print(f"Cycle size: {cycle_size}, t_size: {t}, step: {steps[0]}, proportion: {data}")
                csv["cycle_size"].append(cycle_size)
                csv["t_size"].append(t)
                csv["step"].append(step)
                csv["data_index"].append(data)
                for i in range(0, 16):
                    for l in range(0, 24):
                        csv[f"nat_prob_l{l}_h{i}"].append(hm_nat[l][i] if (len(hm_nat) > 0 and hm_nat[l] is not None) else None)
                        csv[f"icl_prob_l{l}_h{i}"].append(hm_icl[l][i] if (len(hm_icl) > 0 and hm_icl[l] is not None) else None)

    csv = pd.DataFrame(csv)
    # save to csv
    csv.to_csv(Path(path) / "data.csv", index=False)
    print(f"Data saved to {Path(path) / 'data.csv'}")

    

def stats_plot():
    steps = [1, 1000, 5000, 7000, 10000, 100000]
    steps = [f"step{i}" for i in steps]
    cycle_count = list(range(0, 5))
    t_size=[16,32,64,128,256,512]
    layers = list(range(0, 24))
    model_name = "EleutherAI/pythia-1.4b"
    proportions = {"cycle_size": [], "t_size": [], "step": [], "proportion": []}
    # proportion plot 
    for cycle_size in cycle_count:
        for t in t_size:
            for step in steps:
                path = f"/home/mmahaut/projects/parrots/outputs/{model_name}/{step}/"
                data = plot_from_logs(path, cycle_size=cycle_size, t_size=t)
                data= [len(data[i])/10000 for i in range(len(data))]
                proportion=np.mean(data)
                print(f"Cycle size: {cycle_size}, t_size: {t}, step: {step}, proportion: {proportion}")
                proportions["cycle_size"].append(cycle_size)
                proportions["t_size"].append(t)
                proportions["step"].append(step)
                proportions["proportion"].append(proportion)
    proportions = pd.DataFrame(proportions)
    # plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=proportions, x="step", y="proportion", hue="cycle_size", style="t_size")
    plt.title("Proportion of heatmaps")
    plt.xlabel("Step")
    plt.ylabel("Proportion")
    plt.legend(title="Cycle Size")
    plt.savefig(Path(path) / "proportion_plot.png")
    plt.close()
    print(f"Proportion plot saved to {Path(path) / 'proportion_plot.png'}")


# The function below, `proportion_from_previous`, generates Sankey (alluvial/parallel sets) diagrams
# to visualize the flow of repeated and non-repeated elements across training steps.
# It tracks the origin step where each element first appears as repeated, and colors flows accordingly.
# This helps to understand how elements transition between repeated/non-repeated status and their origins.
def proportion_from_previous():

    steps = [1, 1000, 5000, 7000, 10000, 100000]
    steps = [f"step{i}" for i in steps]
    cycle_count = list(range(0, 5))
    t_size = [16, 32, 64, 128, 256, 512]
    model_name = "EleutherAI/pythia-1.4b"

    # Define color palettes for origin tracking (alluvial/parallel sets)
    origin_colorscale = "Viridis"
    
    total = 10000  # Total number of elements to track across steps
    for cycle_size in cycle_count:
        for t in t_size:
            data_per_step = []
            for step in steps:
                path = f"/home/mmahaut/projects/parrots/outputs/{model_name}/{step}/"
                data = plot_from_logs(path, cycle_size=cycle_size, t_size=t)
                data_per_step.append(set(data[0]) if len(data) > 0 else set())

            # For each step, split into repeated and non-repeated
            repeated_per_step = []
            non_repeated_per_step = []
            for step_set in data_per_step:
                repeated = step_set
                non_repeated = set(range(0, total)) - repeated
                repeated_per_step.append(repeated)
                non_repeated_per_step.append(non_repeated)

            # Track the origin of each element (first step where it appears as repeated)
            origins = {}
            for idx in range(total):
                for step_idx, repeated in enumerate(repeated_per_step):
                    if idx in repeated:
                        origins[idx] = step_idx
                        break
                else:
                    origins[idx] = -1  # Never repeated

            # Prepare Sankey diagram nodes: for each step, two nodes (repeated, non-repeated)
            labels = []
            for i, step in enumerate(steps):
                labels.append(f"{step} repeated")
                labels.append(f"{step} non-repeated")

            # For alluvial: assign a color per origin step (or gray for never repeated)
            origin_colors = []
            color_palette = sample_colorscale(origin_colorscale, [i/(len(steps)-1) for i in range(len(steps))])
            for i in range(len(steps)):
                origin_colors.append(color_palette[i])
            never_color = "rgba(180,180,180,0.5)"

            sources = []
            targets = []
            values = []
            link_colors = []

            # For each pair of steps, compute flows between repeated/non-repeated, tracking origins
            for i in range(len(steps) - 1):
                rep_now = repeated_per_step[i]
                nrep_now = non_repeated_per_step[i]
                rep_next = repeated_per_step[i+1]
                nrep_next = non_repeated_per_step[i+1]

                # repeated -> repeated (stayed repeated)
                stayed_repeated = rep_now & rep_next
                # group by origin
                origin_count = {}
                for idx in stayed_repeated:
                    o = origins[idx]
                    # Force all origins at first step to be 0 (step 0)
                    if o > i:
                        o = i
                    origin_count[o] = origin_count.get(o, 0) + 1
                for o, count in origin_count.items():
                    sources.append(i*2)
                    targets.append((i+1)*2)
                    values.append(count)
                    if o == -1:
                        link_colors.append(never_color)
                    else:
                        link_colors.append(origin_colors[o])

                # repeated -> non-repeated (became non-repeated)
                to_non_repeated = rep_now & nrep_next
                origin_count = {}
                for idx in to_non_repeated:
                    o = origins[idx]
                    if o > i:
                        o = i
                    origin_count[o] = origin_count.get(o, 0) + 1
                for o, count in origin_count.items():
                    sources.append(i*2)
                    targets.append((i+1)*2+1)
                    values.append(count)
                    if o == -1:
                        link_colors.append(never_color)
                    else:
                        link_colors.append(origin_colors[o])

                # non-repeated -> repeated (became repeated)
                to_repeated = nrep_now & rep_next
                origin_count = {}
                for idx in to_repeated:
                    o = origins[idx]
                    if o > i:
                        o = i
                    origin_count[o] = origin_count.get(o, 0) + 1
                for o, count in origin_count.items():
                    sources.append(i*2+1)
                    targets.append((i+1)*2)
                    values.append(count)
                    if o == -1:
                        link_colors.append(never_color)
                    else:
                        link_colors.append(origin_colors[o])

                # non-repeated -> non-repeated (stayed non-repeated)
                stayed_non_repeated = nrep_now & nrep_next
                origin_count = {}
                for idx in stayed_non_repeated:
                    o = origins[idx]
                    if o > i:
                        o = i
                    origin_count[o] = origin_count.get(o, 0) + 1
                for o, count in origin_count.items():
                    sources.append(i*2+1)
                    targets.append((i+1)*2+1)
                    values.append(count)
                    if o == -1:
                        link_colors.append(never_color)
                    else:
                        link_colors.append(origin_colors[o])
            
            # reorder so that colors are consistent
            sources = np.array(sources)
            targets = np.array(targets)
            values = np.array(values)
            link_colors = np.array(link_colors)
            # Sort by link_colors (to group flows by origin/color), then by source and target for stability
            sort_indices = np.lexsort((targets, sources, link_colors))
            sources = sources[sort_indices]
            targets = targets[sort_indices]
            values = values[sort_indices]
            link_colors = link_colors[sort_indices]

            # Build the Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=24,
                    line=dict(color="black", width=1.5),
                    label=labels,
                    color=["rgba(44,160,44,0.7)","rgba(200,200,200,0.2)"]*len(steps),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ))])

            fig.update_layout(
                title=dict(
                    text=f"Repeated/Non-repeated Flows with Origins (Alluvial) for cycle_size={cycle_size}, t_size={t}",
                    font=dict(size=22, family="Arial", color="black"),
                    x=0.5,
                ),
                font=dict(size=16, family="Arial", color="black"),
                margin=dict(l=40, r=40, t=80, b=40),
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            out_path = Path(f"/home/mmahaut/projects/parrots/outputs/{model_name}/alluvial_origin_cycle{cycle_size}_t{t}.html")
            fig.write_html(str(out_path))
            print(f"Alluvial/Parallel Sets diagram with origins saved to {out_path}")
            # print the indexes of sentences that started as repeated in the first step, and were repeated in the last step
            first_step_repeated = repeated_per_step[0]
            last_step_repeated = repeated_per_step[-1]
            union = [idx for idx in first_step_repeated if idx in last_step_repeated]
            # save as csv
            union_df = pd.DataFrame({"index": union}) # make sure outpath includes the cycle_size and t_size
            union_df.to_csv(out_path.with_name(out_path.stem + "_repeated_sentences.csv"), index=False)
            print(f"Repeated sentences from first to last step saved to {out_path.with_name(out_path.stem + '_repeated_sentences.csv')}")    


if __name__ == "__main__":
    # stats_plot()
    proportion_from_previous()
    # decentralised_to_csv()

