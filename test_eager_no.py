import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'EleutherAI/pythia-1.4b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, revision='step143000', torch_dtype=torch.float16).to('cuda')
inputs = tokenizer("Hello"*32, return_tensors="pt", truncation=True, max_length=32).to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(len(outputs[0]))
