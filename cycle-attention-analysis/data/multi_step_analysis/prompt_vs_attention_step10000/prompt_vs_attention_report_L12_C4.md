# Prompt vs Attention Analysis Report
**Layer 12, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 74.7% | 10.76 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.59 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.4% | 5.5% | 1.61 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.2% | 0.1% | 0.04 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 11.5% | 0.22 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.0% | 0.01 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.6% | 0.32 | 🔵 **Under-attended** |
| OTHER | 29.9% | 7.3% | 0.24 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 28.8% | 2.14 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 17.3% | 1.95 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.7% | 0.15 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 35.4% | 0.82 | 🟢 **Proportional** |
| BRACKET | 1.3% | 0.3% | 0.24 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 4.3% | 1.21 | 🟡 **Moderately over-attended** |
| OTHER | 24.8% | 13.2% | 0.53 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
