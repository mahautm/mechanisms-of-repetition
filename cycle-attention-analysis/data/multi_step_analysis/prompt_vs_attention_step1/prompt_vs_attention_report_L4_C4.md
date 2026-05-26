# Prompt vs Attention Analysis Report
**Layer 4, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 1.2% | 0.17 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.28 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 0.8% | 0.24 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.7% | 0.22 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 63.7% | 1.19 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.2% | 0.21 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.3% | 0.75 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 31.9% | 1.06 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: CONTENT_WORD
- **Most under-attended**: NEWLINE
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
