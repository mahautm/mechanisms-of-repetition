import torch
import numpy as np
from pathlib import Path

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_results(results, filepath):
    """Save results to file."""
    torch.save(results, filepath)

def load_results(filepath):
    """Load results from file."""
    return torch.load(filepath, map_location='cpu')

def calculate_attention_statistics(attention_matrix):
    """Calculate various statistics for an attention matrix."""
    stats = {}
    
    # Basic statistics
    stats['mean'] = np.mean(attention_matrix)
    stats['std'] = np.std(attention_matrix)
    stats['max'] = np.max(attention_matrix)
    stats['min'] = np.min(attention_matrix)
    
    # Sparsity (percentage of near-zero values)
    threshold = 0.01
    stats['sparsity'] = np.mean(attention_matrix < threshold)
    
    # Concentration (how concentrated the attention is)
    # Higher values mean more concentrated attention
    stats['concentration'] = np.sum(attention_matrix ** 2) / (np.sum(attention_matrix) ** 2) if np.sum(attention_matrix) > 0 else 0
    
    return stats

def format_number(num, decimals=3):
    """Format number for display."""
    if isinstance(num, (int, float)):
        return f"{num:.{decimals}f}"
    return str(num)

def detokenize_tokens(tokenizer, token_ids):
    """
    Convert token IDs back to human-readable text.
    
    Args:
        tokenizer: The tokenizer used for encoding the text.
        token_ids: List of token IDs to be detokenized.
        
    Returns:
        str: Detokenized text.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def compute_attention_change(attention_distributions):
    """
    Compute the change in attention distribution from one cycle to the next for each head.
    
    Args:
        attention_distributions: A list of attention distributions for each head across cycles.
        
    Returns:
        dict: A dictionary with attention head indices as keys and their respective changes as values.
    """
    changes = {}
    for head_idx, distributions in enumerate(attention_distributions):
        if len(distributions) < 2:
            changes[head_idx] = None
            continue
        
        changes[head_idx] = []
        for i in range(1, len(distributions)):
            change = np.abs(distributions[i] - distributions[i - 1])
            changes[head_idx].append(change)
    
    return changes

def normalize_attention_distribution(attention_distribution):
    """
    Normalize the attention distribution for a given head.
    
    Args:
        attention_distribution: The attention distribution to normalize.
        
    Returns:
        np.ndarray: Normalized attention distribution.
    """
    total = np.sum(attention_distribution)
    if total > 0:
        return attention_distribution / total
    return attention_distribution

def aggregate_attention_distributions(attention_distributions):
    """
    Aggregate attention distributions across cycles for each head.
    
    Args:
        attention_distributions: A list of attention distributions for each head across cycles.
        
    Returns:
        np.ndarray: Aggregated attention distribution for each head.
    """
    aggregated = []
    for distributions in attention_distributions:
        aggregated_distribution = np.mean(distributions, axis=0)
        aggregated.append(aggregated_distribution)
    
    return np.array(aggregated)