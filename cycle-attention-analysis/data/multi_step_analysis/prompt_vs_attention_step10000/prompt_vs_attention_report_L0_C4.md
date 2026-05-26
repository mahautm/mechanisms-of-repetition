# Prompt vs Attention Analysis Report
**Layer 0, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 4.7% | 0.68 | 🟡 **Moderately under-attended** |
| TEMPLATE_WORD | 0.5% | 3.1% | 6.11 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 9.3% | 2.69 | 🔴 **Over-attended** |
| PUNCTUATION | 3.2% | 4.1% | 1.26 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 58.3% | 1.09 | 🟢 **Proportional** |
| BRACKET | 0.8% | 0.2% | 0.26 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.9% | 1.06 | 🟢 **Proportional** |
| OTHER | 29.9% | 18.5% | 0.62 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 12.7% | 0.94 | 🟢 **Proportional** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 15.0% | 1.70 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 5.9% | 1.19 | 🟢 **Proportional** |
| CONTENT_WORD | 43.0% | 32.4% | 0.75 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 1.0% | 0.71 | 🟡 **Moderately under-attended** |
| NUMBER | 3.6% | 1.2% | 0.34 | 🔵 **Under-attended** |
| OTHER | 24.8% | 31.8% | 1.28 | 🟡 **Moderately over-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER
