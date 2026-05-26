# Prompt vs Attention Analysis Report
**Layer 20, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 90.8% | 13.21 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.49 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 0.9% | 0.25 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.3% | 0.08 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 3.3% | 0.06 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.25 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.1% | 0.07 | 🔵 **Under-attended** |
| OTHER | 29.6% | 4.2% | 0.14 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: CONTENT_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 30.6% | 5.64 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 8.6% | 1.31 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.3% | 0.85 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 34.8% | 0.67 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.3% | 0.19 | 🔵 **Under-attended** |
| OTHER | 29.8% | 23.3% | 0.78 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
