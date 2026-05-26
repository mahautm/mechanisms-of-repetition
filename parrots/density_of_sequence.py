import pandas as pd
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas()

def get_logits_density(text, model, tokenizer, device):
    """Extract density for each sequence in the DataFrame."""
    toked = tokenizer(text)
    _t = {}
    for k, v in toked.items():
        _t[k] = torch.tensor([v]).to(device)
    o = model(**_t)
    all_logs = o.logits[0, :, :].cpu().detach().numpy()
    selected_logs = all_logs[:, toked["input_ids"][0]]

    log_probs = selected_logs - np.log(np.sum(np.exp(all_logs), axis=1, keepdims=True))
    # sum log probabilities for the entire sequence
    density = np.sum(log_probs) / len(log_probs)

    return density

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results_with_cycles.csv")
    # Initialize the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    
    df['density'] = (df['to_send'] + ' ' + df['generated']).progress_apply(get_logits_density, model=model, tokenizer=tokenizer, device=device)
    # Calculate entropy
    print(df.head())
    # boxplot of density with cycle_count >= 1 and cycle_count == 0
    sns.boxplot(data=df, x="cycle_size", y="density")
    # save the plot
    plt.savefig("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/density_boxplot.png")
    plt.clf()