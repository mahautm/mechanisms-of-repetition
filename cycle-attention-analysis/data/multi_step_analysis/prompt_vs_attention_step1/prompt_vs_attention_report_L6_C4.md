# Prompt vs Attention Analysis Report
**Layer 6, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 1.1% | 0.17 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.09 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 0.9% | 0.25 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.2% | 0.07 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 64.4% | 1.21 | 🟡 **Moderately over-attended** |
| BRACKET | 0.9% | 0.7% | 0.78 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 1.7% | 0.98 | 🟢 **Proportional** |
| OTHER | 30.0% | 31.0% | 1.03 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: CONTENT_WORD
- **Most under-attended**: PUNCTUATION
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
