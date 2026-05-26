import os
import pickle
import numpy as np
from scipy.stats import entropy

import matplotlib.pyplot as plt
import seaborn as sns

def load_pickles(folder_path):
    pickles = []
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pickle'):
            with open(os.path.join(folder_path, filename), 'rb') as f:
                pickles.append(pickle.load(f))
            files.append(filename)
    return pickles, files

def calculate_top_activations(activations, top_n=100):
    top_activations = []
    for layer, layer_activations in activations.items():
        top_indices = np.argsort(layer_activations, axis=0)[-top_n:]
        top_values = layer_activations[top_indices]
        probabilities = top_values / np.sum(layer_activations)
        top_activations.append(probabilities)
    return top_activations

def plot_heatmap(top_activations, output_file):
    plt.figure(figsize=(10, 5))
    sns.heatmap(top_activations, cmap='viridis', cbar=True)
    plt.title('Heatmap of Top 100 Most Activated Dimensions')
    plt.xlabel('Top Dimensions')
    plt.ylabel('Layer')
    plt.savefig(output_file)
    plt.clf()
    plt.close()

def av_distance(activations, baseline_activations):
    distances = []
    for i, layer_activations in activations.items():
        i=int(i)
        distance = 1 - np.dot(layer_activations, baseline_activations[i]) / (np.linalg.norm(layer_activations) * np.linalg.norm(baseline_activations[i]))
        distances.append(distance)
    return distances

def plot_distance(distances, output_file):
    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.title('Average Distance from Baseline')
    plt.xlabel('Layer')
    plt.ylabel('Distance')
    plt.savefig(output_file)
    plt.clf()
    plt.close()

def main(folder_path):
    pickles, filenames = load_pickles(folder_path)
    baseline_index = filenames.index('hlayers.pickle')
    baseline_activations = pickles[baseline_index]
    for i, activations in enumerate(pickles):
        top_activations = calculate_top_activations(activations)
        plot_heatmap(top_activations, os.path.join(folder_path, filenames[i].replace('.pickle', '.png')))
        distances = av_distance(pickles[i], baseline_activations)
        plot_distance(distances, os.path.join(folder_path, filenames[i].replace('.pickle', '_distance.png')))

if __name__ == "__main__":
    folder_path = '/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/hlayers'
    main(folder_path)
