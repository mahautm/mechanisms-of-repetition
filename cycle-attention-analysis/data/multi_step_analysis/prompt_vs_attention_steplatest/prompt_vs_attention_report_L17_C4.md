# Prompt vs Attention Analysis Report
**Layer 17, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 93.9% | 14.17 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.28 | 🔵 **Under-attended** |
| SENTENCE_END | 3.6% | 0.1% | 0.04 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.1% | 0.02 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 3.7% | 0.07 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.0% | 0.02 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.2% | 0.14 | 🔵 **Under-attended** |
| OTHER | 28.6% | 1.9% | 0.07 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 42.9% | 4.19 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.15 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.2% | 0.76 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 2.0% | 0.1% | 0.04 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 30.4% | 0.83 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.1% | 0.03 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.2% | 0.81 | 🟢 **Proportional** |
| OTHER | 40.0% | 20.1% | 0.50 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
