from datasets import load_dataset

def load_text_dataset(dataset_name="JeanKaddour/minipile", split="train", seed=42, n_samples=5000):
    dataset = load_dataset(dataset_name)
    subset = dataset[split].shuffle(seed=seed)
    subset = subset.select(range(0, n_samples))
    return subset["text"]

def pretokenize_texts(texts, tokenizer, max_length=256):
    return [tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length, padding_side="left") for t in texts]
