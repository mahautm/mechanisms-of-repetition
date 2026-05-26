# Prompt vs Attention Analysis Report
**Layer 16, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 91.7% | 13.22 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 1.0% | 0.28 | 🔵 **Under-attended** |
| PUNCTUATION | 3.2% | 0.1% | 0.04 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 3.0% | 0.06 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.1% | 0.07 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.0% | 0.01 | 🔵 **Under-attended** |
| OTHER | 29.9% | 4.1% | 0.14 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 37.0% | 2.75 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 14.5% | 1.64 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.2% | 0.04 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 31.4% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.0% | 0.01 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 4.2% | 1.17 | 🟢 **Proportional** |
| OTHER | 24.8% | 12.8% | 0.52 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
