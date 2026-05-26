# Prompt vs Attention Analysis Report
**Layer 14, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 84.9% | 12.34 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.61 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.4% | 0.4% | 0.13 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.2% | 0.05 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 6.0% | 0.11 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.6% | 0.72 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.5% | 0.31 | 🔵 **Under-attended** |
| OTHER | 29.6% | 7.1% | 0.24 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 28.0% | 5.15 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.0% | 1.37 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.4% | 0.89 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 36.5% | 0.70 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.3% | 0.20 | 🔵 **Under-attended** |
| OTHER | 29.8% | 23.7% | 0.79 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
