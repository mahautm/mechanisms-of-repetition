import sys
sys.path.append('/home/mmahaut/projects/parrots')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from parrots.aa_fortu.modules.data_utils import load_text_dataset
from parrots.cycle_detection import detect_cycles

def generate_and_detect(model, tokenizer, text, max_new_tokens=1000, max_length=32, device='cuda'):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    generated_ids = outputs[0].tolist()
    cycle, cycle_size, cycle_count = detect_cycles(generated_ids)
    is_repeating = cycle_count > 0
    return is_repeating, cycle_size, cycle_count, len(generated_ids)

model_name = 'EleutherAI/pythia-1.4b'
revision = 'steplatest' # or whatever
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(model_name, revision='step143000', torch_dtype=torch.float16).to(device)

texts = load_text_dataset(seed=42, n_samples=5)
for i, text in enumerate(texts):
    print(f"Sample {i}:")
    is_rep, sz, c, ln = generate_and_detect(model, tokenizer, text)
    print(f"  Natural: rep={is_rep}, sz={sz}, c={c}, len={ln}")
