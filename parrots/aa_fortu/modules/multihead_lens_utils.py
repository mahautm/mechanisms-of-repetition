# Utility functions for loading multi-head lenses
import torch
import torch.nn as nn
from pathlib import Path

class MultiHeadLens(nn.Module):
    """Separate lenses for each attention head"""
    def __init__(self, head_dim, vocab_size, num_heads=16):
        super(MultiHeadLens, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        
        # Create separate lens for each head
        self.head_lenses = nn.ModuleList([
            nn.Linear(head_dim, vocab_size) for _ in range(num_heads)
        ])
    
    def forward(self, x, head_idx=None):
        """
        x: tensor of shape [batch, seq, head_dim] for single head
        or [batch, seq, num_heads, head_dim] for all heads
        head_idx: which head to apply lens to (if x is single head)
        """
        if head_idx is not None:
            # Single head input
            return self.head_lenses[head_idx](x)
        else:
            # Multi-head input - apply each lens to its corresponding head
            if len(x.shape) == 4:  # [batch, seq, num_heads, head_dim]
                outputs = []
                for h in range(self.num_heads):
                    head_output = self.head_lenses[h](x[:, :, h, :])  # [batch, seq, vocab_size]
                    outputs.append(head_output)
                return torch.stack(outputs, dim=2)  # [batch, seq, num_heads, vocab_size]
            elif len(x.shape) == 3 and x.shape[-1] == self.head_dim:
                # Single head: [batch, seq, head_dim]
                if head_idx is None:
                    raise ValueError("head_idx must be specified for single head input")
                return self.head_lenses[head_idx](x)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

def load_multihead_lens(lens_path, device="cuda"):
    """Load a multi-head lens from file"""
    lens_path = Path(lens_path)
    
    if not lens_path.exists():
        raise FileNotFoundError(f"Lens file not found: {lens_path}")
    
    # Load the saved lens data
    lens_data = torch.load(lens_path, map_location=device)
    
    # Extract parameters
    num_heads = lens_data['num_heads']
    head_dim = lens_data['head_dim']
    vocab_size = lens_data['vocab_size']
    
    # Create the lens and load state dict
    multihead_lens = MultiHeadLens(head_dim, vocab_size, num_heads)
    multihead_lens.load_state_dict(lens_data['state_dict'])
    multihead_lens.to(device)
    
    print(f"Loaded multi-head lens: {num_heads} heads, {head_dim}D -> {vocab_size}D")
    
    return multihead_lens

def load_lens_for_analysis(lens_path, device="cuda"):
    """Load lens for contrast analysis - returns either single lens or multi-head lens"""
    if lens_path is None:
        return None
    
    lens_path = Path(lens_path)
    
    # Check if it's a multi-head lens file
    if "multihead" in lens_path.name:
        return load_multihead_lens(lens_path, device)
    else:
        # Load regular single lens
        lens = torch.load(lens_path, map_location=device)
        lens.to(device)
        print(f"Loaded single lens from {lens_path}")
        return lens

# Example usage:
# lens = load_lens_for_analysis("lenses/EleutherAI_pythia-1.4b/layer_1_multihead_lens.pth")