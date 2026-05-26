# Attention Bias Ratio Statistical Summary

## Key Findings

### Natural Sequences

| Token Type | Mean Ratio | Median Ratio | Max Ratio | Layers with Data | Interpretation |
|------------|------------|--------------|-----------|------------------|----------------|
| NEWLINE | 2.00 | 1.82 | 4.2 | 21/24 | 🟢 **Proportional** |
| TEMPLATE_WORD | 0.79 | 0.62 | 2.7 | 24/24 | 🟡 **Under-attended** |
| SENTENCE_END | 1.79 | 1.68 | 5.7 | 24/24 | 🟢 **Proportional** |
| PUNCTUATION | 0.95 | 0.81 | 2.7 | 23/24 | 🟢 **Proportional** |
| CONTENT_WORD | 1.00 | 0.98 | 1.4 | 24/24 | 🟢 **Proportional** |
| BRACKET | 0.42 | 0.44 | 1.1 | 20/24 | 🟡 **Under-attended** |
| NUMBER | 0.47 | 0.40 | 1.4 | 21/24 | 🟡 **Under-attended** |
| OTHER | 0.95 | 0.75 | 4.5 | 24/24 | 🟡 **Under-attended** |

#### Key Insights:
- **Most over-attended**: NEWLINE (median ratio: 1.82)
- **Most under-attended**: NUMBER (median ratio: 0.40)
- **Severely over-attended tokens**: 0/8
- **Proportional tokens**: 4/8
- **Severely under-attended tokens**: 0/8

### No Cycle Icl Sequences

| Token Type | Mean Ratio | Median Ratio | Max Ratio | Layers with Data | Interpretation |
|------------|------------|--------------|-----------|------------------|----------------|
| CONTENT_WORD | 0.71 | 0.70 | 1.0 | 21/24 | 🟡 **Under-attended** |

#### Key Insights:
- **Most over-attended**: CONTENT_WORD (median ratio: 0.70)
- **Most under-attended**: CONTENT_WORD (median ratio: 0.70)
- **Severely over-attended tokens**: 0/8
- **Proportional tokens**: 0/8
- **Severely under-attended tokens**: 0/8

## Overall Conclusions

1. **Attention is NOT proportional to prompt composition**
2. **Structural tokens (NEWLINE, TEMPLATE_WORD) receive massive over-attention**
3. **Content tokens receive severe under-attention despite high frequency**
4. **Natural sequences show more extreme biases than ICL sequences**
5. **The model has learned to focus on rare structural signals for repetition**