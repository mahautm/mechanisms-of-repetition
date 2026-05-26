# Prompt vs Attention Analysis Report
**Layer 20, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 92.6% | 13.35 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.04 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 0.9% | 0.26 | 🔵 **Under-attended** |
| PUNCTUATION | 3.2% | 0.1% | 0.03 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 2.5% | 0.05 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.0% | 0.02 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.1% | 0.06 | 🔵 **Under-attended** |
| OTHER | 29.9% | 3.7% | 0.13 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 39.4% | 2.93 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 13.8% | 1.56 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 30.2% | 0.70 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 3.3% | 0.92 | 🟢 **Proportional** |
| OTHER | 24.8% | 13.4% | 0.54 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
