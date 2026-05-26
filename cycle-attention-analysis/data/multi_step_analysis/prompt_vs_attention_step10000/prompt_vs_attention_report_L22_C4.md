# Prompt vs Attention Analysis Report
**Layer 22, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 86.5% | 12.46 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.44 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 1.3% | 0.38 | 🔵 **Under-attended** |
| PUNCTUATION | 3.2% | 0.3% | 0.08 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 7.0% | 0.13 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.1% | 0.09 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.2% | 0.12 | 🔵 **Under-attended** |
| OTHER | 29.9% | 4.4% | 0.15 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 36.0% | 2.68 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 11.4% | 1.29 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.8% | 0.17 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 33.6% | 0.78 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.1% | 0.09 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 4.4% | 1.22 | 🟡 **Moderately over-attended** |
| OTHER | 24.8% | 13.7% | 0.55 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
