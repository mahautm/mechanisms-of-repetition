# Prompt vs Attention Analysis Report
**Layer 5, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 61.3% | 8.83 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 1.1% | 2.10 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 4.6% | 1.34 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.2% | 1.1% | 0.35 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 22.7% | 0.42 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.0% | 0.05 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.1% | 0.65 | 🟡 **Moderately under-attended** |
| OTHER | 29.9% | 8.1% | 0.27 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 32.6% | 2.42 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 17.3% | 1.96 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 1.1% | 0.23 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 30.7% | 0.72 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.6% | 0.45 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 2.9% | 0.82 | 🟢 **Proportional** |
| OTHER | 24.8% | 14.6% | 0.59 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
