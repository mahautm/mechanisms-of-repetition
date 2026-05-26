# Prompt vs Attention Analysis Report
**Layer 23, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 40.7% | 5.92 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 2.4% | 4.85 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 3.0% | 0.87 | 🟢 **Proportional** |
| PUNCTUATION | 3.3% | 0.7% | 0.22 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 11.3% | 0.21 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 1.9% | 2.32 | 🔴 **Over-attended** |
| NUMBER | 1.8% | 2.0% | 1.15 | 🟢 **Proportional** |
| OTHER | 29.6% | 38.0% | 1.29 | 🟡 **Moderately over-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: CONTENT_WORD
- **High attention categories** (>20%): NEWLINE, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 15.2% | 2.80 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 8.7% | 1.32 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.6% | 0.96 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 40.5% | 0.78 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.1% | 0.04 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 1.7% | 1.02 | 🟢 **Proportional** |
| OTHER | 29.8% | 31.3% | 1.05 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER
