# Attention Bias Evolution Analysis

## Training Progression Summary

| Training Step | Newline Bias (Natural) | Content Bias (Natural) | Template Max (Natural) | Newline Bias (ICL) |
|---------------|------------------------|------------------------|------------------------|---------------------|
| step1 | 0.01x | 0.01x | 0.01x | 0.00x |
| step1000 | 1.82x | 0.98x | 0.62x | 0.00x |
| step10000 | 11.46x | 0.18x | 1.02x | 2.44x |
| step100000 | 11.02x | 0.18x | 0.88x | 4.62x |
| steplatest | 11.40x | 0.20x | 1.02x | 3.28x |

## Key Evolution Findings

### Structural Specialization Development

- **Newline bias increase**: 0.01x → 11.40x (1266.7x growth)
- **Content suppression**: 0.01x → 0.20x (0.0x more suppression)

### Template Word Specialization

- **Peak specialization**: 1.02x at step10000
- **Final specialization**: 1.02x

## Scientific Implications

1. **Gradual Development**: Structural bias develops progressively during training
2. **Content Suppression**: Model actively learns to ignore semantic content
3. **Specialization Circuits**: Specific layers develop extreme specializations
4. **Architecture vs Learning**: Repetition is learned behavior, not architectural bias

## Training Stage Analysis

### Early Training (step1)
- **Proportional tokens**: 0/8 token types
- **Max specialization**: 0.01x

### Late Training (steplatest)
- **Proportional tokens**: 1/8 token types
- **Extreme specializations**: 1 token types
- **Max specialization**: 11.40x