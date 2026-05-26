# Train separate lenses for each attention head
import torch
import torch.nn as nn
from parrots.aa_fortu.ckpt_pipeline_2 import HookedModel
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm
from threading import Lock
import typer

last_layer = {
    "mistralai/Mistral-7B-Instruct-v0.3":"model.norm",
    "mistralai/Mistral-7B-v0.3":"model.norm",
    "EleutherAI/pythia-1.4b":"final_layer_norm",
    "allenai/OLMo-1B-hf":"model.norm",
    "allenai/OLMo-1B-0724-hf":"model.norm",
}

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

    @classmethod
    def from_dict(cls, _dict):
        """Instantiate from a state dict"""
        num_heads = _dict['num_heads']
        head_dim = _dict['head_dim']
        vocab_size = _dict['vocab_size']
        model = cls(head_dim, vocab_size, num_heads)
        model.load_state_dict(_dict['state_dict'])
        return model
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load to handle head lenses"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception as e:
                    raise RuntimeError(f"While copying the parameter named {name}, "
                                       f"whose dimensions in the model are {own_state[name].size()} and "
                                       f"whose dimensions in the checkpoint are {param.size()}.") from e
            elif strict:
                raise KeyError(f'unexpected key "{name}" in state_dict')
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(f'missing keys in state_dict: "{missing}"')
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

def train_multihead_attention_lens(
    model_name: str = "EleutherAI/pythia-1.4b",
    revision: str = None,
    epochs: int = 10,
    lr: float = 1e-3,
    train_samples: int = 10000,
    layer_idx: int = 1,
    output_dir: str = "lenses",
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 512,
    num_heads: int = 16,
    head_dim: int = 128,
):
    """Train separate lenses for each attention head of a specific layer"""
    
    # Load model and tokenizer
    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Create hooked model for the specific layer
    # Determine layer naming based on model architecture
    if "pythia" in model_name.lower():
        layer_name = f"gpt_neox.layers.{layer_idx}"
    elif "olmo" in model_name.lower() or "mistral" in model_name.lower():
        layer_name = f"model.layers.{layer_idx}"
    else:
        # Default to model.layers for most modern architectures
        layer_name = f"model.layers.{layer_idx}"
    
    hooked_model = HookedModel(model, layer=f"{layer_name}.attention")
    
    # Initialize multi-head lens
    vocab_size = model.config.vocab_size
    multihead_lens = MultiHeadLens(head_dim, vocab_size, num_heads).to(device)
    
    # Optimizer for all head lenses
    optimizer = torch.optim.Adam(multihead_lens.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Load training data
    print("Loading training data...")
    dataset = load_dataset("JeanKaddour/minipile", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42).take(train_samples)
    
    # Training loop
    print(f"Training {num_heads} separate lenses for layer {layer_idx}")
    
    total_loss = 0
    num_batches = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        epoch_batches = 0
        
        # Process data in batches
        batch_texts = []
        for sample in tqdm(dataset, desc=f"Epoch {epoch + 1}"):
            text = sample['text']
            if len(text) > 50:  # Skip very short texts
                batch_texts.append(text)
                
                if len(batch_texts) >= batch_size:
                    # Process batch
                    batch_loss = train_batch(
                        batch_texts, hooked_model, multihead_lens, 
                        tokenizer, optimizer, criterion,
                        layer_idx, num_heads, head_dim, device, max_length
                    )
                    
                    if batch_loss is not None:
                        epoch_loss += batch_loss
                        epoch_batches += 1
                    
                    batch_texts = []
        
        # Process remaining texts
        if batch_texts:
            batch_loss = train_batch(
                batch_texts, hooked_model, multihead_lens,
                tokenizer, optimizer, criterion,
                layer_idx, num_heads, head_dim, device, max_length
            )
            if batch_loss is not None:
                epoch_loss += batch_loss
                epoch_batches += 1
        
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.6f}")
    
    # Save the trained lenses
    output_path = Path(output_dir) / model_name.replace("/", "_") / f"layer_{layer_idx}_multihead_lens.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_name': model_name,
        'layer_idx': layer_idx,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'vocab_size': vocab_size,
        'state_dict': multihead_lens.state_dict(),
    }, output_path)
    
    print(f"Saved multi-head lens to {output_path}")
    return output_path

def train_batch(batch_texts, hooked_model, multihead_lens, tokenizer, optimizer, criterion,
                layer_idx, num_heads, head_dim, device, max_length):
    """Train on a single batch"""
    
    try:
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            # Forward pass to get attention outputs and final logits
            outputs = hooked_model.model(**inputs)
            final_logits = outputs.logits  # [batch, seq, vocab]
            
            # Get attention outputs
            if not hooked_model.attn_outputs:
                return None
                
            layer_name, attention_data = hooked_model.attn_outputs[-1]
            
            # Extract attention tensor
            if isinstance(attention_data, (list, tuple)):
                attention_tensor = attention_data[0]  # Take first element (attention output)
            else:
                attention_tensor = attention_data
                
            # Split into heads: [batch, seq, 2048] -> [batch, seq, 16, 128]
            batch_size, seq_len, hidden_size = attention_tensor.shape
            attention_heads = attention_tensor.reshape(batch_size, seq_len, num_heads, head_dim)
            
        # Clear attention outputs for next batch
        hooked_model.clear()
        
        # Train each head lens
        optimizer.zero_grad()
        total_batch_loss = 0
        
        for head_idx in range(num_heads):
            # Get head-specific attention: [batch, seq, head_dim]
            head_attention = attention_heads[:, :, head_idx, :]
            
            # Apply head lens: [batch, seq, head_dim] -> [batch, seq, vocab]
            head_logits = multihead_lens(head_attention, head_idx=head_idx)
            
            # Target: final model logits
            target_probs = torch.nn.functional.log_softmax(final_logits, dim=-1)
            head_probs = torch.nn.functional.log_softmax(head_logits, dim=-1)
            
            # KL divergence loss
            head_loss = criterion(head_probs.view(-1, head_probs.size(-1)), 
                                target_probs.view(-1, target_probs.size(-1)).exp())
            
            total_batch_loss += head_loss
        
        # Backward pass
        total_batch_loss.backward()
        optimizer.step()
        
        return total_batch_loss.item()
        
    except Exception as e:
        print(f"Error in batch training: {e}")
        return None

def main(
    model_name: str = "EleutherAI/pythia-1.4b",
    revision: str = None,
    epochs: int = 10,
    lr: float = 1e-3,
    train_samples: int = 10000,
    layer_idx: int = 1,
    output_dir: str = "lenses",
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 512,
    num_heads: int = 16,
    head_dim: int = 128,
):
    """Main training function"""
    
    # Auto-detect model configuration if not specified
    if num_heads == 16 and head_dim == 128:  # Default values
        if "pythia" in model_name:
            num_heads = 16
            head_dim = 128
        elif "qwen" in model_name.lower():
            # Add other model configurations as needed
            num_heads = 32  # Example for larger models
            head_dim = 128
        # Add more model-specific configurations here
    
    output_path = train_multihead_attention_lens(
        model_name=model_name,
        revision=revision,
        epochs=epochs,
        lr=lr,
        train_samples=train_samples,
        layer_idx=layer_idx,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    
    print(f"Training complete! Saved to: {output_path}")

if __name__ == "__main__":
    typer.run(main)