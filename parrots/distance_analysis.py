# from a matrix recording distances shaped [sequences, tokens], plot distance increase

def plot_distance_increase(dists, output_file):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))
    # reorder each line to have increasing distance
    dists = np.sort(dists, axis=1)
    # plot average distance increase for each token in the vocabulary, averaged over all sequences, with a 95% confidence interval
    # the x-axis is the token index in the vocabulary, the y-axis is the average distance increase
    sns.lineplot(data=dists, dashes=False, ax=ax, ci=95)
    ax.set_xlabel("Token")
    ax.set_ylabel("Average distance increase")
    plt.savefig(output_file)

if __name__ == "__main__":
    df = pd.DataFrame({"text": ["The capital of France is", "The capital of Germany is", "The capital of Italy is"]})
    dists = get_distance("facebook/opt-1.3b", df)
    plot_distance_increase(dists, "distance_increase.png") 