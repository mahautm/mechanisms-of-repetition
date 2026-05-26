# Prompt vs Attention Analysis Report
**Layer 23, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 63.8% | 9.20 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 1.1% | 2.25 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 3.6% | 1.04 | 🟢 **Proportional** |
| PUNCTUATION | 3.2% | 0.9% | 0.29 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 21.0% | 0.39 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.21 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.1% | 0.63 | 🟡 **Moderately under-attended** |
| OTHER | 29.9% | 8.3% | 0.28 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 29.9% | 2.23 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 8.8% | 13.2% | 1.49 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.9% | 0.17 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 34.1% | 0.79 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.1% | 0.08 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 4.8% | 1.36 | 🟡 **Moderately over-attended** |
| OTHER | 24.8% | 16.9% | 0.68 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
