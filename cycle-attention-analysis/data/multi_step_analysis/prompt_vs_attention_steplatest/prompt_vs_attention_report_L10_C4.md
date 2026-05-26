# Prompt vs Attention Analysis Report
**Layer 10, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 73.7% | 11.12 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.50 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.6% | 1.2% | 0.35 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 1.1% | 0.32 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 13.7% | 0.25 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.3% | 0.42 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.9% | 0.49 | 🔵 **Under-attended** |
| OTHER | 28.6% | 8.9% | 0.31 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: CONTENT_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 31.7% | 3.09 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.9% | 0.89 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.1% | 0.06 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 36.1% | 0.99 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.3% | 0.13 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 3.1% | 1.13 | 🟢 **Proportional** |
| OTHER | 40.0% | 23.8% | 0.60 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
