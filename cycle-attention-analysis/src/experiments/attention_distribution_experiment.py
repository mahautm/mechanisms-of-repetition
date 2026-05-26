import os
import numpy as np
import torch
from pathlib import Path
from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from parrots.aa_fortu.modules.attention_analyzer import AttentionAnalyzer
from parrots.aa_fortu.modules.cycle_processor import CycleProcessor
from parrots.aa_fortu.modules.visualization import plot_attention_distribution
from parrots.aa_fortu.modules.utils import load_text_dataset

def run_attention_distribution_experiment(
    model_name: str = "EleutherAI/pythia-1.4b",
    n_cycles: int = 5,
    batch_size: int = 1,
    max_length: int = 256,
    n_samples: int = 5000,
    output_dir: str = "data/results",
    plot_dir: str = "plots"
):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = get_device()
    model.to(device)

    # Load and preprocess data
    texts = load_text_dataset(n_samples=n_samples)
    
    # Initialize CycleProcessor and analyze cycles
    cycle_processor = CycleProcessor()
    cycles = cycle_processor.process_cycles(texts, n_cycles=n_cycles)

    # Initialize AttentionAnalyzer
    attention_analyzer = AttentionAnalyzer(model, tokenizer, device)

    # Analyze attention distribution for each cycle
    attention_results = {}
    for cycle_idx, cycle in enumerate(cycles):
        attention_distribution = attention_analyzer.analyze_attention(cycle, max_length=max_length)
        attention_results[cycle_idx] = attention_distribution

    # Detokenize and save results
    os.makedirs(output_dir, exist_ok=True)
    for cycle_idx, distribution in attention_results.items():
        detokenized_words = attention_analyzer.detokenize_attention(distribution)
        np.save(os.path.join(output_dir, f"attention_cycle_{cycle_idx}.npy"), detokenized_words)

    # Plot attention distributions
    os.makedirs(plot_dir, exist_ok=True)
    for cycle_idx, distribution in attention_results.items():
        plot_attention_distribution(distribution, title=f"Attention Distribution for Cycle {cycle_idx}", save_path=os.path.join(plot_dir, f"attention_distribution_cycle_{cycle_idx}.png"))

if __name__ == "__main__":
    run_attention_distribution_experiment()