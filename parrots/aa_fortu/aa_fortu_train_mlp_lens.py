# Train lenses for MLP blocks
import torch
import torch.nn as nn
from parrots.aa_fortu.modules.model_utils import HookedModel, load_model_and_tokenizer, get_device
from pathlib import Path
from datasets import load_dataset
import re
from tqdm import tqdm
import typer

last_layer = {
    "mistralai/Mistral-7B-Instruct-v0.3":"model.norm",
    "mistralai/Mistral-7B-v0.3":"model.norm",
    "EleutherAI/pythia-1.4b":"final_layer_norm",
}

class MLPLens(nn.Module):
    """Single lens for MLP output"""
    def __init__(self, mlp_dim, vocab_size):
        super(MLPLens, self).__init__()
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        
        # Create lens for MLP output
        self.lens = nn.Linear(mlp_dim, vocab_size)
    
    def forward(self, x):
        """
        x: tensor of shape [batch, seq, mlp_dim] for MLP output
        """
        return self.lens(x)

def get_mlp_dim_for_model(model_name):
    """Get MLP output dimension for different model architectures"""
    # MLP output dimension is the same as hidden_size (not intermediate_size)
    # The intermediate dimension (4x hidden_size) is internal to the MLP
    if "pythia-1.4b" in model_name:
        return 2048  # hidden_size for Pythia-1.4B
    elif "pythia-2.8b" in model_name:
        return 2560  # hidden_size for Pythia-2.8B  
    elif "pythia-6.9b" in model_name:
        return 4096  # hidden_size for Pythia-6.9B
    elif "pythia-12b" in model_name:
        return 5120  # hidden_size for Pythia-12B
    # TODO: Add other model architectures
    # Mistral-7B: 4096 (hidden_size)
    # LLaMA models vary by size
    else:
        # Fallback: common Pythia sizes
        if "pythia" in model_name.lower():
            if "410m" in model_name:
                return 1024  # hidden_size for Pythia-410M
            elif "1b" in model_name:
                return 1280  # hidden_size for Pythia-1B
            elif "160m" in model_name:
                return 768   # hidden_size for Pythia-160M
            elif "70m" in model_name:
                return 512   # hidden_size for Pythia-70M
        
        # Final fallback - try to get from model config
        raise ValueError(f"Unknown model MLP dimension for {model_name}. Please add to get_mlp_dim_for_model()")

def train_mlp_lens(
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
):
    """Train lens for MLP output of a specific layer"""
    
    # Load model and tokenizer
    print(f"Loading model {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, revision, use_bfloat16=True)
    model.eval()
    device = get_device()
    model.to(device)
    
    # Create hooked model for the specific MLP layer
    if "pythia" in model_name.lower():
        layer_name = f"gpt_neox.layers.{layer_idx}.mlp"
    else:
        # TODO: Add MLP layer names for other model architectures
        # For Mistral: f"model.layers.{layer_idx}.mlp"
        # For LLaMA: f"model.layers.{layer_idx}.mlp"
        raise NotImplementedError(f"MLP analysis not yet implemented for {model_name}")
    
    hooked_model = HookedModel(model, layer=layer_name)
    
    # Get actual MLP dimension from a forward pass
    print("Determining MLP dimension from model...")
    vocab_size = model.config.vocab_size
    
    # Do a quick forward pass to get the actual MLP dimension
    dummy_input = torch.randint(0, vocab_size, (1, 10)).to(device)
    with torch.no_grad():
        _ = hooked_model(dummy_input)
    
    mlp_outputs = hooked_model.get_mlp_outputs()
    if not mlp_outputs:
        raise RuntimeError(f"No MLP outputs captured for layer {layer_idx}")
    
    # Get the actual MLP dimension from the tensor shape
    _, sample_output = mlp_outputs[0]  # Take first (should be only one for single layer)
    if len(sample_output.shape) == 3:
        _, _, actual_mlp_dim = sample_output.shape
    else:
        raise RuntimeError(f"Unexpected MLP output shape: {sample_output.shape}")
    
    print(f"Actual MLP dimension: {actual_mlp_dim}")
    hooked_model.clear()  # Clear the test outputs
    
    # Initialize MLP lens with actual dimension
    mlp_lens = MLPLens(actual_mlp_dim, vocab_size).to(device)
    
    # Match model dtype (BFloat16)
    if model.dtype == torch.bfloat16:
        mlp_lens = mlp_lens.to(torch.bfloat16)
    
    # Optimizer for MLP lens
    optimizer = torch.optim.Adam(mlp_lens.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Load training data
    print("Loading training data...")
    dataset = load_dataset("JeanKaddour/minipile", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42).take(train_samples)
    
    # Training loop
    print(f"Training MLP lens for layer {layer_idx}")
    
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
                        batch_texts, hooked_model, mlp_lens, 
                        tokenizer, optimizer, criterion,
                        layer_idx, device, max_length
                    )
                    
                    if batch_loss is not None:
                        epoch_loss += batch_loss
                        epoch_batches += 1
                    
                    batch_texts = []
        
        # Process remaining texts
        if batch_texts:
            batch_loss = train_batch(
                batch_texts, hooked_model, mlp_lens,
                tokenizer, optimizer, criterion,
                layer_idx, device, max_length
            )
            if batch_loss is not None:
                epoch_loss += batch_loss
                epoch_batches += 1
        
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.6f}")
    
    # Save the trained lens
    output_path = Path(output_dir) / model_name.replace("/", "_") / f"layer_{layer_idx}_mlp_lens.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_name': model_name,
        'layer_idx': layer_idx,
        'mlp_dim': actual_mlp_dim,
        'vocab_size': vocab_size,
        'state_dict': mlp_lens.state_dict(),
        'lens_type': 'mlp'
    }, output_path)
    
    print(f"Saved MLP lens to {output_path}")
    return output_path

def train_batch(batch_texts, hooked_model, mlp_lens, tokenizer, optimizer, criterion,
                layer_idx, device, max_length):
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
            # Forward pass to get MLP outputs and final logits
            outputs = hooked_model.model(**inputs)
            final_logits = outputs.logits  # [batch, seq, vocab]
            
            # Get MLP outputs - use the new MLP-specific method
            if not hooked_model.has_mlp_outputs():
                return None
                
            mlp_outputs = hooked_model.get_mlp_outputs()
            if not mlp_outputs:
                return None
                
            layer_name, mlp_data = mlp_outputs[-1]
            
            # Extract MLP tensor
            if isinstance(mlp_data, (list, tuple)):
                mlp_tensor = mlp_data[0]  # Take first element (MLP output)
            else:
                mlp_tensor = mlp_data
                
            # MLP output shape: [batch, seq, mlp_dim]
            batch_size, seq_len, mlp_dim = mlp_tensor.shape
            
        # Clear outputs for next batch
        hooked_model.clear()
        
        # Train MLP lens
        optimizer.zero_grad()
        
        # Ensure MLP tensor matches lens dtype
        if mlp_lens.lens.weight.dtype != mlp_tensor.dtype:
            mlp_tensor = mlp_tensor.to(mlp_lens.lens.weight.dtype)
        
        # Apply MLP lens: [batch, seq, mlp_dim] -> [batch, seq, vocab]
        mlp_logits = mlp_lens(mlp_tensor)
        
        # Ensure target logits match for loss computation
        if final_logits.dtype != mlp_logits.dtype:
            final_logits = final_logits.to(mlp_logits.dtype)
        
        # Target: final model logits
        target_probs = torch.nn.functional.log_softmax(final_logits, dim=-1)
        mlp_probs = torch.nn.functional.log_softmax(mlp_logits, dim=-1)
        
        # KL divergence loss
        loss = criterion(mlp_probs.view(-1, mlp_probs.size(-1)), 
                        target_probs.view(-1, target_probs.size(-1)).exp())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
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
):
    """Main training function for MLP lens"""
    
    output_path = train_mlp_lens(
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
    )
    
    print(f"MLP lens training complete! Saved to: {output_path}")

if __name__ == "__main__":
    typer.run(main)