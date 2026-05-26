# Prompt vs Attention Analysis Report
**Layer 19, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 85.8% | 12.48 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.49 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 0.6% | 0.17 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.2% | 0.07 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 6.8% | 0.13 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.29 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.2% | 0.14 | 🔵 **Under-attended** |
| OTHER | 29.6% | 5.9% | 0.20 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 28.8% | 5.29 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 8.9% | 1.35 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.4% | 0.87 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 35.5% | 0.68 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.1% | 0.06 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.2% | 0.13 | 🔵 **Under-attended** |
| OTHER | 29.8% | 24.1% | 0.81 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
