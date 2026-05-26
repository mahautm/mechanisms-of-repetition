import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "swiss-ai/Apertus-8B-2509"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

print(f"Inputs: {input_ids}")

outputs = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=False
)

print(f"Outputs 1: {outputs}")
print("Decoded 1:", tokenizer.batch_decode(outputs, skip_special_tokens=False))

print("Done.")
