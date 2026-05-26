# Prompt vs Attention Analysis Report
**Layer 6, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 74.2% | 11.20 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.44 | 🔵 **Under-attended** |
| SENTENCE_END | 3.6% | 1.7% | 0.48 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 1.1% | 0.32 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 15.3% | 0.28 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.2% | 0.28 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.9% | 0.54 | 🟡 **Moderately under-attended** |
| OTHER | 28.6% | 6.3% | 0.22 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: OTHER
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 31.7% | 3.09 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 5.6% | 1.01 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.7% | 0.36 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 34.1% | 0.93 | 🟢 **Proportional** |
| BRACKET | 2.5% | 1.4% | 0.55 | 🟡 **Moderately under-attended** |
| NUMBER | 2.8% | 2.7% | 0.99 | 🟢 **Proportional** |
| OTHER | 40.0% | 23.8% | 0.60 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
