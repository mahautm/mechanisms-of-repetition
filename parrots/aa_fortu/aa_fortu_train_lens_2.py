# as in tuned lens, we train an affine transformation to minimize KL divergence between intermediary and output distribution
import torch
import torch.nn as nn
# from transformer_lens import HookedTransformer
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
}
class Lens(nn.Module):
    def __init__(self, embed_size):
        super(Lens, self).__init__()
        self.lens = nn.Linear(embed_size, embed_size)
        self.bias = nn.Parameter(torch.zeros(embed_size))
    
    def forward(self, x):
        return self.lens(x) + self.bias

class MultiHeadLens(nn.Module):
    """Lens for individual attention heads"""
    def __init__(self, head_dim, vocab_size, num_heads=16):
        super(MultiHeadLens, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Create separate lens for each head
        self.head_lenses = nn.ModuleList([
            nn.Linear(head_dim, vocab_size) for _ in range(num_heads)
        ])
    
    def forward(self, x, head_idx=None):
        """
        x: tensor of shape [batch, head_dim] for single head
        or [batch, seq, num_heads, head_dim] for all heads
        head_idx: which head to apply lens to (if x is single head)
        """
        if head_idx is not None:
            # Single head input
            return self.head_lenses[head_idx](x)
        else:
            # Multi-head input - apply each lens to its corresponding head
            batch_size, seq_len = x.shape[:2]
            if len(x.shape) == 4:  # [batch, seq, num_heads, head_dim]
                outputs = []
                for h in range(self.num_heads):
                    head_output = self.head_lenses[h](x[:, :, h, :])  # [batch, seq, vocab_size]
                    outputs.append(head_output)
                return torch.stack(outputs, dim=2)  # [batch, seq, num_heads, vocab_size]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

def train_attention_lens(
    # model_name:str="EleutherAI/pythia-1.4b",
    model_name:str="Qwen/Qwen2.5-7B",
    revision:str=None,
    epochs:int=10,
    batch_size:int=1,
    lr:float=0.001,
    layer_idx:int=10,
    save_path:str=None,
    seed:int=42,
    do_bfloat16:bool=False,
):
    
    if save_path is None:
        save_path = f"./lenses/{model_name}" if revision is None else f"./lenses/{model_name}_{revision}"
    # given hooked places in the transformer, align using kl-loss and linear lens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # LOAD model using lens
    probed_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, device_map="auto")
    probed_model.eval()
    # probed_model.to(device)
    hooked_model = HookedModel(probed_model, layer=layer_idx)
    hooked_model.add_hooks([last_layer[model_name]])
    tokenizer_lock = Lock()
    with tokenizer_lock:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probed_model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Using model {model_name} with revision {revision} and layer {layer_idx}")
    print(f"Model has {len(hooked_model.hooks)} layers, using layer {layer_idx} for lensing")
    # deactivate gradients
    for param in hooked_model.parameters():
        param.requires_grad = False
    hooked_model.eval()

    # DATA --> part of the pile that we are not using for the repetition analyisis
    dataset = load_dataset("JeanKaddour/minipile")
    subset = dataset["train"].shuffle(seed=seed)
    subset = subset.select(range(len(subset) - 10000, len(subset))) # only keep lat 10k datapoints
    # only keep 50 first words in every text
    subset = subset.map(lambda x: {"text": x["text"][:50]})
    print(f"Training on {len(subset)} samples")


    lenses={}
    optimizers={}
    for epoch in range(epochs):
        for batch_start_idx in tqdm(range(0, len(subset), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            text_batch=subset.select(range(batch_start_idx, batch_start_idx + batch_size))["text"]
            with torch.no_grad():
                torch.cuda.empty_cache()
                toked_batch= tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
                hooked_model(**toked_batch)
            # print(f"hooked_model.attn_outputs keys: {hooked_model.attn_outputs[-1]}")
            cache = {k: v for k, v in hooked_model.attn_outputs}
            final_o = cache.pop(last_layer[model_name]).to(device)
            for k,v in cache.items():
                if k not in lenses:
                    tqdm.write(f"Creating lens for {k}")
                    lenses[k]=Lens(v[0].shape[-1]).to(device).to(torch.bfloat16 if do_bfloat16 else torch.float32)
                    lenses[k].train()
                    optimizers[k]=torch.optim.Adam(lenses[k].parameters(), lr=lr)
                lens_o = lenses[k](v[0])
                lens_o_log_softmax = lens_o.log_softmax(dim=-1)
                final_o_softmax = final_o.expand_as(lens_o).log_softmax(dim=-1)
                loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(lens_o_log_softmax, final_o_softmax)
                tqdm.write(f"{k} Loss: {loss.item()}")
                loss.backward()
                optimizers[k].step()
                optimizers[k].zero_grad()


    # save
    Path(save_path).mkdir(exist_ok=True, parents=True)
    for k,v in lenses.items():
        torch.save(v, Path(save_path)/(k+".pth"))

if __name__ == "__main__":
    typer.run(train_attention_lens)




