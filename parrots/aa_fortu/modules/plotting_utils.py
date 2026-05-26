import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(heatmap, title, subtitle, xlabel, ylabel, save_path, vmin=0, vmax=0.2):
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, cmap="viridis", cbar=True, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.suptitle(subtitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()
