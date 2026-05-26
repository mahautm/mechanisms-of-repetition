# Prompt vs Attention Analysis Report
**Layer 4, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 43.3% | 6.24 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 1.1% | 2.30 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 6.4% | 1.87 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.2% | 1.9% | 0.58 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 33.4% | 0.62 | 🟡 **Moderately under-attended** |
| BRACKET | 0.8% | 0.0% | 0.02 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.0% | 0.59 | 🟡 **Moderately under-attended** |
| OTHER | 29.9% | 12.8% | 0.43 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 26.8% | 1.99 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 13.0% | 1.47 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 4.6% | 0.92 | 🟢 **Proportional** |
| CONTENT_WORD | 43.0% | 36.3% | 0.84 | 🟢 **Proportional** |
| BRACKET | 1.3% | 2.0% | 1.50 | 🟡 **Moderately over-attended** |
| NUMBER | 3.6% | 1.3% | 0.37 | 🔵 **Under-attended** |
| OTHER | 24.8% | 16.1% | 0.65 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
