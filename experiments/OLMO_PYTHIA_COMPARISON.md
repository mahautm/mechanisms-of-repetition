# OLMo vs Pythia Architecture Comparison

## Model Specifications

### OLMo-1B-hf
- **Hidden size**: 2048
- **Num layers**: 16
- **Num attention heads**: 16  
- **Vocab size**: 50304
- **Max position embeddings**: 2048
- **Hook pattern**: `model.layers.{layer_idx}`
- **Layer structure**: OlmoDecoderLayer → OlmoAttention

### Pythia-1.4b
- **Hidden size**: 2048
- **Num layers**: 24
- **Num attention heads**: 16
- **Vocab size**: 50304
- **Max position embeddings**: 2048
- **Hook pattern**: `gpt_neox.layers.{layer_idx}`
- **Layer structure**: GPTNeoXLayer → GPTNeoXAttention

## Key Differences

1. **Layer Count**: 
   - OLMo-1B: 16 layers
   - Pythia-1.4b: 24 layers
   - **Proportional mapping**: Layer 12/16 (OLMo) ≈ Layer 19/24 (Pythia) = 75% depth

2. **Hook Path**:
   - OLMo: `model.layers.{layer_idx}`
   - Pythia: `gpt_neox.layers.{layer_idx}`

3. **Architecture Names**:
   - OLMo: OlmoDecoderLayer, OlmoAttention, OlmoMLP
   - Pythia: GPTNeoXLayer, GPTNeoXAttention, GPTNeoXMLP

## Adaptation Requirements

### For `aa_fortu.py` / `ckpt_pipeline_main.py`

Change hook path from:
```python
hooked_model = HookedModel(model, layer=f"gpt_neox.layers.{str(single_lens)}")
```

To:
```python
hooked_model = HookedModel(model, layer=f"model.layers.{str(single_lens)}")
```

### For alluvial analysis

Target layer selection:
- Pythia layer 19/24 = 79.2% depth
- OLMo equivalent: 0.792 × 16 ≈ **layer 12** (or 13)

### No Checkpoint Evolution Available

OLMo is released as final model only, not intermediate checkpoints.
**Alternative approach**: Compare OLMo-1B vs OLMo-7B as different model sizes (not training steps)
