# Prompt vs Attention Analysis Report
**Layer 5, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 70.9% | 10.31 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.56 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.4% | 2.5% | 0.73 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 0.8% | 0.25 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 14.9% | 0.28 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.4% | 0.54 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.4% | 0.21 | 🔵 **Under-attended** |
| OTHER | 29.6% | 9.8% | 0.33 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 23.7% | 4.35 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.1% | 1.38 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.4% | 0.89 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 37.7% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.1% | 0.07 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.6% | 0.35 | 🔵 **Under-attended** |
| OTHER | 29.8% | 26.5% | 0.89 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
