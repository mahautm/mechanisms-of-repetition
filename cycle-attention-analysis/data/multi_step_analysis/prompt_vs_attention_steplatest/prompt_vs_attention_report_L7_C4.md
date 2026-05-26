# Prompt vs Attention Analysis Report
**Layer 7, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 61.4% | 9.26 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.6% | 1.15 | 🟢 **Proportional** |
| SENTENCE_END | 3.6% | 5.3% | 1.49 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.5% | 1.4% | 0.41 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 17.0% | 0.31 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.3% | 0.40 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.6% | 0.93 | 🟢 **Proportional** |
| OTHER | 28.6% | 12.4% | 0.43 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: CONTENT_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 28.3% | 2.76 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.20 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 7.1% | 1.29 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.0% | 0.9% | 0.47 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 34.1% | 0.93 | 🟢 **Proportional** |
| BRACKET | 2.5% | 2.3% | 0.90 | 🟢 **Proportional** |
| NUMBER | 2.8% | 2.8% | 1.02 | 🟢 **Proportional** |
| OTHER | 40.0% | 24.4% | 0.61 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
