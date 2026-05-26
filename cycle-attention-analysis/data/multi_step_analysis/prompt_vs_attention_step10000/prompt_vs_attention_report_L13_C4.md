# Prompt vs Attention Analysis Report
**Layer 13, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 85.7% | 12.35 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.23 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 4.3% | 1.24 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.2% | 0.1% | 0.02 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 4.7% | 0.09 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.2% | 0.11 | 🔵 **Under-attended** |
| OTHER | 29.9% | 5.0% | 0.17 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 35.3% | 2.62 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 18.4% | 2.09 | 🔴 **Over-attended** |
| PUNCTUATION | 5.0% | 0.5% | 0.10 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 30.7% | 0.72 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.1% | 0.11 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 3.8% | 1.06 | 🟢 **Proportional** |
| OTHER | 24.8% | 11.1% | 0.45 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
