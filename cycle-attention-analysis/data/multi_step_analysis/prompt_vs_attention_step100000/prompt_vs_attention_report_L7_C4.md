# Prompt vs Attention Analysis Report
**Layer 7, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 58.5% | 8.50 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.80 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 6.5% | 1.90 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 1.9% | 0.59 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.8% | 19.4% | 0.36 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.9% | 1.15 | 🟢 **Proportional** |
| NUMBER | 1.8% | 1.0% | 0.55 | 🟡 **Moderately under-attended** |
| OTHER | 29.6% | 11.3% | 0.38 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: CONTENT_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 20.8% | 3.82 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.7% | 1.47 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 3.9% | 1.44 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 51.9% | 37.0% | 0.71 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.9% | 0.56 | 🟡 **Moderately under-attended** |
| NUMBER | 1.6% | 0.3% | 0.21 | 🔵 **Under-attended** |
| OTHER | 29.8% | 27.4% | 0.92 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
