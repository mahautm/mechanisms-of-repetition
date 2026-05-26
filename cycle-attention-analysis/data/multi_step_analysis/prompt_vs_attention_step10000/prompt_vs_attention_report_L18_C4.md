# Prompt vs Attention Analysis Report
**Layer 18, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 90.7% | 13.07 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.08 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 0.4% | 0.12 | 🔵 **Under-attended** |
| PUNCTUATION | 3.2% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 3.8% | 0.07 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.1% | 0.17 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.0% | 0.03 | 🔵 **Under-attended** |
| OTHER | 29.9% | 4.9% | 0.16 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 35.4% | 2.63 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 14.9% | 1.69 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.0% | 0.01 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 32.4% | 0.75 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.0% | 0.02 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 3.8% | 1.07 | 🟢 **Proportional** |
| OTHER | 24.8% | 13.3% | 0.54 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
