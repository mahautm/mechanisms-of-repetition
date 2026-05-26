# Prompt vs Attention Analysis Report
**Layer 12, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 76.9% | 11.61 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.62 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.6% | 0.8% | 0.23 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.7% | 0.20 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 15.0% | 0.27 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.1% | 0.17 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.9% | 0.49 | 🔵 **Under-attended** |
| OTHER | 28.6% | 5.3% | 0.19 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 33.5% | 3.27 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.15 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 5.5% | 1.00 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.4% | 0.19 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 33.8% | 0.93 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.6% | 0.23 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.3% | 0.83 | 🟢 **Proportional** |
| OTHER | 40.0% | 23.9% | 0.60 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
