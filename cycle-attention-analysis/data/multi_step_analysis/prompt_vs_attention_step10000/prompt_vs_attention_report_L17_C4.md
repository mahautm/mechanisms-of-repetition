# Prompt vs Attention Analysis Report
**Layer 17, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 92.7% | 13.36 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.17 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 0.8% | 0.23 | 🔵 **Under-attended** |
| PUNCTUATION | 3.2% | 0.2% | 0.06 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 2.3% | 0.04 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.1% | 0.15 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.1% | 0.04 | 🔵 **Under-attended** |
| OTHER | 29.9% | 3.8% | 0.13 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 35.0% | 2.60 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 14.4% | 1.63 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.2% | 0.03 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 33.2% | 0.77 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.0% | 0.02 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 4.1% | 1.13 | 🟢 **Proportional** |
| OTHER | 24.8% | 13.1% | 0.53 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
