#!/usr/bin/env python3
"""
Inspect OLMo model structure to understand correct hook paths
"""
import sys
sys.path.append('/home/mmahaut/projects/parrots')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def inspect_model(model_name="allenai/OLMo-1B-hf"):
    print(f"Inspecting {model_name}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Model config
    print("\n📊 Model Configuration:")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Num layers: {model.config.num_hidden_layers}")
    print(f"  - Num attention heads: {model.config.num_attention_heads}")
    print(f"  - Vocab size: {model.config.vocab_size}")
    print(f"  - Max position embeddings: {model.config.max_position_embeddings}")
    
    # Model structure
    print("\n🏗️  Model Structure (top-level):")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
    
    # Layer structure
    print("\n🔍 Layer Structure (first 3 layers):")
    for i, (name, module) in enumerate(model.named_modules()):
        if i > 20:  # Limit output
            print("  ...")
            break
        if 'layer' in name.lower() or 'block' in name.lower():
            print(f"  - {name}: {type(module).__name__}")
    
    # Attention module paths
    print("\n🎯 Attention Module Paths:")
    attention_paths = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            attention_paths.append(name)
            if len(attention_paths) <= 5:
                print(f"  - {name}: {type(module).__name__}")
    
    if len(attention_paths) > 5:
        print(f"  ... ({len(attention_paths)} total attention modules)")
    
    # Find layer pattern
    print("\n🔎 Detecting Layer Pattern:")
    layer_names = [name for name, _ in model.named_modules() if 'layers.' in name or 'blocks.' in name]
    if layer_names:
        example = layer_names[0]
        print(f"  Example path: {example}")
        
        # Extract pattern
        if 'model.layers.' in example:
            print(f"  ✓ Hook pattern: 'model.layers.{{layer_idx}}'")
        elif 'transformer.blocks.' in example:
            print(f"  ✓ Hook pattern: 'transformer.blocks.{{layer_idx}}'")
        elif 'gpt_neox.layers.' in example:
            print(f"  ✓ Hook pattern: 'gpt_neox.layers.{{layer_idx}}'")
        else:
            print(f"  ⚠️  Unknown pattern - manual inspection needed")
    
    # Test tokenization
    print("\n🔤 Tokenizer Test:")
    test_text = "The capital of France is"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"  Input: '{test_text}'")
    print(f"  Tokens: {tokens['input_ids'][0].tolist()}")
    print(f"  Decoded: {tokenizer.decode(tokens['input_ids'][0])}")
    
    # Model forward pass test
    print("\n⚡ Forward Pass Test:")
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        print(f"  Logits shape: {logits.shape}")
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        print(f"  Next token: {next_token} ({tokenizer.decode([next_token])})")
    
    print("\n" + "=" * 80)
    print("✓ Inspection complete!")
    
    return model

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMo-1B-hf"
    inspect_model(model_name)
