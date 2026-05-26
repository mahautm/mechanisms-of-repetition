# Prompt vs Attention Analysis Report
**Layer 14, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 87.3% | 12.58 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.25 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 2.8% | 0.81 | 🟢 **Proportional** |
| PUNCTUATION | 3.2% | 0.1% | 0.02 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 5.3% | 0.10 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.1% | 0.05 | 🔵 **Under-attended** |
| OTHER | 29.9% | 4.4% | 0.15 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 33.1% | 2.46 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 15.8% | 1.79 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.3% | 0.06 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 33.2% | 0.77 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 4.5% | 1.27 | 🟡 **Moderately over-attended** |
| OTHER | 24.8% | 13.1% | 0.53 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
