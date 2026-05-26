# Prompt vs Attention Analysis Report
**Layer 19, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 86.5% | 12.46 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.36 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 1.8% | 0.52 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.2% | 0.0% | 0.02 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 6.5% | 0.12 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.0% | 0.02 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.2% | 0.13 | 🔵 **Under-attended** |
| OTHER | 29.9% | 4.8% | 0.16 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 36.5% | 2.71 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 14.3% | 1.62 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.6% | 0.12 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 31.6% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.1% | 0.09 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 3.9% | 1.08 | 🟢 **Proportional** |
| OTHER | 24.8% | 13.0% | 0.52 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
