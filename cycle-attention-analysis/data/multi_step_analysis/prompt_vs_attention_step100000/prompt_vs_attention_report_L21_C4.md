# Prompt vs Attention Analysis Report
**Layer 21, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 78.4% | 11.40 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 1.09 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 1.8% | 0.51 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 0.5% | 0.14 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 8.1% | 0.15 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.6% | 0.73 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.9% | 0.49 | 🔵 **Under-attended** |
| OTHER | 29.6% | 9.3% | 0.31 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 27.4% | 5.04 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.18 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.4% | 1.42 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.6% | 0.95 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 36.3% | 0.70 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.1% | 0.09 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.5% | 0.29 | 🔵 **Under-attended** |
| OTHER | 29.8% | 23.7% | 0.79 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
