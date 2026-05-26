# Prompt vs Attention Analysis Report
**Layer 6, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 51.5% | 7.42 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.88 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 8.0% | 2.31 | 🔴 **Over-attended** |
| PUNCTUATION | 3.2% | 2.2% | 0.69 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 25.1% | 0.47 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.24 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.9% | 1.09 | 🟢 **Proportional** |
| OTHER | 29.9% | 10.7% | 0.36 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 24.9% | 1.85 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 14.1% | 1.59 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 3.1% | 0.62 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 43.0% | 37.0% | 0.86 | 🟢 **Proportional** |
| BRACKET | 1.3% | 2.4% | 1.80 | 🟡 **Moderately over-attended** |
| NUMBER | 3.6% | 2.9% | 0.83 | 🟢 **Proportional** |
| OTHER | 24.8% | 15.5% | 0.62 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
