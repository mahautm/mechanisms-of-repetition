# as in tuned lens, we train an affine transformation to minimize KL divergence between intermediary and output distribution
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm
from threading import Lock
import typer

class Lens(nn.Module):
    def __init__(self, embed_size):
        super(Lens, self).__init__()
        self.lens = nn.Linear(embed_size, embed_size)
        self.bias = nn.Parameter(torch.zeros(embed_size))
    
    def forward(self, x):
        return self.lens(x) + self.bias

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
    # device = torch.device("cpu")
    print("Device:", device)
    probed_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)

    tokenizer_lock = Lock()
    with tokenizer_lock:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probed_model.config.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        hooked_model = HookedTransformer.from_pretrained(
            model_name,
            revision=revision,
            device=device,
            hf_model=probed_model,
            tokenizer=tokenizer,
            n_embd=probed_model.config.hidden_size,
            n_layer=probed_model.config.num_hidden_layers,
            n_head=probed_model.config.num_attention_heads,
            vocab_size=probed_model.config.vocab_size,
            n_ctx=probed_model.config.max_position_embeddings,
            dtype=torch.bfloat16 if do_bfloat16 else None,
        )
    # deactivate gradients
    for param in hooked_model.parameters():
        param.requires_grad = False
    hooked_model.eval()
    del probed_model
    hooked_model.set_use_attn_result(True)
    def hook_site(name: str):
        if name.endswith("hook_result"):
            block_idx = int(re.search(r"\.(\d+)\.", name).group(1))
            if block_idx == layer_idx:
                return True
        elif name.endswith("ln_final.hook_normalized"):
            return True
        return False
    cache = hooked_model.add_caching_hooks(hook_site)

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
                toked_batch=hooked_model.to_tokens(text_batch)
                hooked_model(toked_batch, return_type=None)
            final_o = cache.pop("ln_final.hook_normalized")
            for k,v in cache.items():
                if k not in lenses:
                    tqdm.write(f"Creating lens for {k}")
                    lenses[k]=Lens(cache[k].shape[-1]).to(device).to(torch.bfloat16 if do_bfloat16 else torch.float32)
                    lenses[k].train()
                    optimizers[k]=torch.optim.Adam(lenses[k].parameters(), lr=lr)

                optimizers[k].zero_grad()
                lens_o = lenses[k](v)
                lens_o_log_softmax = lens_o.log_softmax(dim=-1)
                final_o_softmax = final_o.unsqueeze(2).expand_as(lens_o).log_softmax(dim=-1)
                loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(lens_o_log_softmax, final_o_softmax)
                tqdm.write(f"{k} Loss: {loss.item()}")
                loss.backward()
                optimizers[k].step()

    # save
    Path(save_path).mkdir(exist_ok=True, parents=True)
    for k,v in lenses.items():
        torch.save(v, Path(save_path)/(k+".pth"))

if __name__ == "__main__":
    typer.run(train_attention_lens)




