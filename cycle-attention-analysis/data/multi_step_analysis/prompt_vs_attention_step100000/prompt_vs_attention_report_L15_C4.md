# Prompt vs Attention Analysis Report
**Layer 15, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 84.6% | 12.31 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.37 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 0.6% | 0.17 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.4% | 0.12 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 7.2% | 0.13 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.3% | 0.42 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.3% | 0.19 | 🔵 **Under-attended** |
| OTHER | 29.6% | 6.4% | 0.22 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 29.7% | 5.46 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 8.3% | 1.27 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.3% | 0.83 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 35.7% | 0.69 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.0% | 0.01 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.3% | 0.21 | 🔵 **Under-attended** |
| OTHER | 29.8% | 23.7% | 0.79 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
