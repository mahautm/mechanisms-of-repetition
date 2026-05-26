# Prompt vs Attention Analysis Report
**Layer 13, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 86.7% | 12.61 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.54 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.4% | 0.8% | 0.22 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.3% | 0.08 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 6.0% | 0.11 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.3% | 0.40 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.6% | 0.37 | 🔵 **Under-attended** |
| OTHER | 29.6% | 5.1% | 0.17 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 29.3% | 5.39 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 10.2% | 1.55 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.6% | 0.94 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 35.1% | 0.68 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.3% | 0.16 | 🔵 **Under-attended** |
| OTHER | 29.8% | 22.6% | 0.76 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
