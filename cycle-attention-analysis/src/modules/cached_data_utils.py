from datasets import load_dataset, load_from_disk
from pathlib import Path
import os

def load_cached_dataset(dataset_name="JeanKaddour/minipile", split="train", seed=42, n_samples=200):
    """Load dataset from cache to avoid API rate limits."""
    
    # Try to load from cache first
    cache_dir = Path("../data/dataset_cache")
    
    try:
        if cache_dir.exists():
            print(f"Attempting to load dataset from cache...")
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(cache_dir)
            )
        else:
            print("No cache found, loading from Hugging Face...")
            dataset = load_dataset(dataset_name)
            
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        print("Trying fallback dataset (wikitext)...")
        
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            dataset_name = "wikitext"
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("Using dummy data for testing...")
            return generate_dummy_texts(n_samples)
    
    # Handle different dataset structures
    if dataset_name == "wikitext":
        texts = [text for text in dataset[split]["text"] if text.strip()]
    else:
        # For minipile and similar datasets
        texts = dataset[split]["text"]
    
    # Sample the data
    if len(texts) > n_samples:
        import random
        random.seed(seed)
        texts = random.sample(texts, n_samples)
    
    # Filter out very short texts
    texts = [text for text in texts if len(text.strip()) > 50]
    
    print(f"Loaded {len(texts)} text samples")
    return texts

def generate_dummy_texts(n_samples=200):
    """Generate dummy repetitive texts for testing when datasets fail."""
    print("Generating dummy repetitive texts for testing...")
    
    base_patterns = [
        "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
        "Hello world, how are you? Hello world, how are you? Hello world, how are you?",
        "Python is great for data science. Python is great for data science. Python is great for data science.",
        "Machine learning models learn patterns. Machine learning models learn patterns. Machine learning models learn patterns.",
        "Attention mechanisms focus on relevant parts. Attention mechanisms focus on relevant parts. Attention mechanisms focus on relevant parts."
    ]
    
    texts = []
    for i in range(n_samples):
        # Create longer texts with repetitive patterns
        pattern = base_patterns[i % len(base_patterns)]
        extended_text = " ".join([pattern] * 3)  # Repeat 3 times
        extended_text += f" This is sample number {i+1}. " + pattern
        texts.append(extended_text)
    
    return texts

def load_text_dataset(dataset_name="JeanKaddour/minipile", split="train", seed=42, n_samples=200):
    """Wrapper function to maintain compatibility."""
    return load_cached_dataset(dataset_name, split, seed, n_samples)