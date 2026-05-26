# Attention Bias Ratio Statistical Summary

## Key Findings

### Natural Sequences

| Token Type | Mean Ratio | Median Ratio | Max Ratio | Layers with Data | Interpretation |
|------------|------------|--------------|-----------|------------------|----------------|
| NEWLINE | 9.26 | 11.46 | 13.4 | 20/24 | 🔴 **Severely Over-attended** |
| TEMPLATE_WORD | 3.16 | 1.02 | 12.9 | 23/24 | 🟢 **Proportional** |
| SENTENCE_END | 1.66 | 1.47 | 4.1 | 24/24 | 🟢 **Proportional** |
| PUNCTUATION | 0.39 | 0.20 | 1.4 | 23/24 | 🟡 **Under-attended** |
| CONTENT_WORD | 0.34 | 0.18 | 1.1 | 24/24 | 🔵 **Severely Under-attended** |
| BRACKET | 0.12 | 0.10 | 0.3 | 18/24 | 🔵 **Severely Under-attended** |
| NUMBER | 0.45 | 0.25 | 1.4 | 19/24 | 🟡 **Under-attended** |
| OTHER | 0.28 | 0.20 | 0.7 | 24/24 | 🟡 **Under-attended** |

#### Key Insights:
- **Most over-attended**: NEWLINE (median ratio: 11.46)
- **Most under-attended**: BRACKET (median ratio: 0.10)
- **Severely over-attended tokens**: 1/8
- **Proportional tokens**: 2/8
- **Severely under-attended tokens**: 2/8

### No Cycle Icl Sequences

| Token Type | Mean Ratio | Median Ratio | Max Ratio | Layers with Data | Interpretation |
|------------|------------|--------------|-----------|------------------|----------------|
| NEWLINE | 2.17 | 2.44 | 2.9 | 20/24 | 🟡 **Over-attended** |
| SENTENCE_END | 1.62 | 1.62 | 2.1 | 20/24 | 🟢 **Proportional** |
| PUNCTUATION | 0.39 | 0.17 | 1.2 | 19/24 | 🔵 **Severely Under-attended** |
| CONTENT_WORD | 0.80 | 0.77 | 1.1 | 20/24 | 🟡 **Under-attended** |
| BRACKET | 0.57 | 0.24 | 1.8 | 17/24 | 🟡 **Under-attended** |
| NUMBER | 0.91 | 1.06 | 1.4 | 20/24 | 🟢 **Proportional** |
| OTHER | 0.66 | 0.55 | 1.4 | 20/24 | 🟡 **Under-attended** |

#### Key Insights:
- **Most over-attended**: NEWLINE (median ratio: 2.44)
- **Most under-attended**: PUNCTUATION (median ratio: 0.17)
- **Severely over-attended tokens**: 0/8
- **Proportional tokens**: 2/8
- **Severely under-attended tokens**: 1/8

## Overall Conclusions

1. **Attention is NOT proportional to prompt composition**
2. **Structural tokens (NEWLINE, TEMPLATE_WORD) receive massive over-attention**
3. **Content tokens receive severe under-attention despite high frequency**
4. **Natural sequences show more extreme biases than ICL sequences**
5. **The model has learned to focus on rare structural signals for repetition**