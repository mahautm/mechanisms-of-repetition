# Prompt vs Attention Analysis Report
**Layer 22, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 79.3% | 11.96 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 0.96 | 🟢 **Proportional** |
| SENTENCE_END | 3.6% | 1.0% | 0.29 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.2% | 0.06 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 9.5% | 0.17 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.3% | 0.41 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.7% | 0.39 | 🔵 **Under-attended** |
| OTHER | 28.6% | 8.6% | 0.30 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 37.6% | 3.67 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.40 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.4% | 0.80 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 2.0% | 0.2% | 0.08 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 32.2% | 0.88 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.5% | 0.18 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.2% | 0.81 | 🟢 **Proportional** |
| OTHER | 40.0% | 22.8% | 0.57 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
