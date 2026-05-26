# Prompt vs Attention Analysis Report
**Layer 11, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 72.6% | 10.96 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.79 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.6% | 1.2% | 0.33 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.8% | 0.23 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 12.4% | 0.23 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.2% | 0.31 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.2% | 0.71 | 🟡 **Moderately under-attended** |
| OTHER | 28.6% | 11.1% | 0.39 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: CONTENT_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 33.7% | 3.28 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.12 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 5.0% | 0.91 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.3% | 0.14 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 34.0% | 0.93 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.6% | 0.23 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.3% | 0.84 | 🟢 **Proportional** |
| OTHER | 40.0% | 24.1% | 0.60 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
