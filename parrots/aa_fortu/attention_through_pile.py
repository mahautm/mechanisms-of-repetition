
# from Michael Han Inversion view, modified by matéo mahaut (add link to original)

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import torch
import nltk
import re
import itertools
from tqdm import tqdm
import json
from pathlib import Path
torch.set_grad_enabled(False)
# nltk.download('punkt_tab')

dataset = load_dataset("JeanKaddour/minipile")
subset = dataset["train"].shuffle(seed=0)
subset = subset.select(range(len(subset)//10))

def split_sents(examples):
    sents = []
    for text in examples["text"]:
        sents.extend( list(filter(lambda x: len(x) <= 100, nltk.sent_tokenize(text))) )
    return {"sentence": sents}

print(subset)
subset = subset.map(split_sents, batched=True, remove_columns=subset.column_names, num_proc=4)
print(subset)

# model_name = "facebook/opt-1.3b"
model_name = "EleutherAI/pythia-1.4B"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True)
probed_model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
probed_model.resize_token_embeddings(probed_model.config.vocab_size+1)
probed_model.config.pad_token_id = tokenizer.pad_token_id

hooked_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=probed_model,
        tokenizer=tokenizer,
        # n_embd=probed_model.config.n_embd,
        # n_layer=probed_model.config.n_layer,
        # n_head=probed_model.config.n_head,
        # vocab_size=probed_model.config.vocab_size,
        # n_ctx=probed_model.config.n_positions,
        n_embd=probed_model.config.hidden_size,
        n_layer=probed_model.config.num_hidden_layers,
        n_head=probed_model.config.num_attention_heads,
        vocab_size=probed_model.config.vocab_size,
        n_ctx=probed_model.config.max_position_embeddings,
)

hooked_model.eval()
del probed_model

def hook_site(name: str):
    if name.endswith("hook_pattern"):
        return True
    return False

cache = hooked_model.add_caching_hooks(hook_site)


bz = 64
dir_data_path = "/home/mmahaut/projects/exps/parr/EleutherAI/pythia-1.4B/rraa/average_selected_heads.json"
with open(dir_data_path, "r") as f:
    dir_heads_active_freq = json.load(f)
selected_heads = [k for k, v in sorted(dir_heads_active_freq.items(), key=lambda x: x[1], reverse=True)[:25]]
# random select heads
# selected_heads = random.sample(list(dir_heads_active_freq.keys()), 25)
selected_heads = " ".join(selected_heads)
print(selected_heads)

text_per_head = {h: [] for h in selected_heads.split(" ")}

try:
    for i in tqdm(range(0, len(subset), bz)):
        text_batch = subset.select(range(i, min(i+bz, len(subset))))["sentence"]
        tokens_batch = hooked_model.to_tokens(text_batch)
        assert (tokens_batch[:, 0] == tokenizer.bos_token_id).all()
        hooked_model(tokens_batch, return_type=None)

        pad_mask = tokens_batch == tokenizer.pad_token_id

        for h in text_per_head:
            block_idx, head_idx = h.split(".")
            block_idx, head_idx = int(block_idx), int(head_idx)
            # [batch, head_index, query_pos, key_pos]
            attn_on_bos = cache[f"blocks.{block_idx}.attn.hook_pattern"][:, head_idx, :, 0].contiguous()
            attn_on_bos.masked_fill_(pad_mask, 1.0)
            mask = (attn_on_bos.min(dim=1)[0] < 0.6).tolist()

            text_per_head[h].extend( [t for t, m in zip(text_batch, mask) if m] )

except:
    print("WARNING: error raised")

text_per_head = DatasetDict( {h: Dataset.from_dict({"sentence": t}) for h, t in text_per_head.items()} )
 

outpath = "./outputs/factual/pile_text_per_head"

# mkdir if not exist
Path(outpath).mkdir(parents=True, exist_ok=True)
# save
text_per_head.save_to_disk(outpath)