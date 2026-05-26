# Prompt vs Attention Analysis Report
**Layer 20, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 3.3% | 0.49 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.04 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 1.0% | 0.29 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.4% | 0.11 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 58.7% | 1.10 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.3% | 0.29 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.7% | 0.99 | 🟢 **Proportional** |
| OTHER | 30.0% | 34.6% | 1.15 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: OTHER
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
