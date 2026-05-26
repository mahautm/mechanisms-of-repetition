# Prompt vs Attention Analysis Report
**Layer 15, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 0.9% | 0.13 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.11 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 0.7% | 0.20 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.5% | 0.14 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 63.9% | 1.20 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.5% | 0.62 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 1.3% | 0.74 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 32.1% | 1.07 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: CONTENT_WORD
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: OTHER
