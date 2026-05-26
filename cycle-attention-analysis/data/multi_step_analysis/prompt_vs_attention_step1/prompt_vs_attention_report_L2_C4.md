# Prompt vs Attention Analysis Report
**Layer 2, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 0.6% | 0.09 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.13 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 0.8% | 0.23 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.8% | 0.24 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 63.1% | 1.18 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.1% | 0.08 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.8% | 1.03 | 🟢 **Proportional** |
| OTHER | 30.0% | 32.7% | 1.09 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: CONTENT_WORD
- **Most under-attended**: BRACKET
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
