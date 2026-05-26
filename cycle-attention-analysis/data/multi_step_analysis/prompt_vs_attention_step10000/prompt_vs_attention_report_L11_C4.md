# Prompt vs Attention Analysis Report
**Layer 11, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 71.3% | 10.28 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 1.02 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 5.5% | 1.60 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.2% | 0.9% | 0.26 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 12.6% | 0.23 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.23 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.4% | 0.25 | 🔵 **Under-attended** |
| OTHER | 29.9% | 8.6% | 0.29 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 30.3% | 2.26 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 17.1% | 1.94 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.9% | 0.18 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 36.6% | 0.85 | 🟢 **Proportional** |
| BRACKET | 1.3% | 0.9% | 0.71 | 🟡 **Moderately under-attended** |
| NUMBER | 3.6% | 3.4% | 0.95 | 🟢 **Proportional** |
| OTHER | 24.8% | 10.6% | 0.43 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
