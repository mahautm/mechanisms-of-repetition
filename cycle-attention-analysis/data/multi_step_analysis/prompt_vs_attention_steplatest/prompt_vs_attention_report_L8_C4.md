# Prompt vs Attention Analysis Report
**Layer 8, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 58.7% | 8.86 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 1.00 | 🟢 **Proportional** |
| SENTENCE_END | 3.6% | 4.7% | 1.32 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.5% | 1.9% | 0.55 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 54.8% | 22.8% | 0.42 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.1% | 0.13 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.3% | 0.73 | 🟡 **Moderately under-attended** |
| OTHER | 28.6% | 10.0% | 0.35 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 30.2% | 2.95 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.06 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 6.7% | 1.22 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.0% | 0.9% | 0.43 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 32.8% | 0.90 | 🟢 **Proportional** |
| BRACKET | 2.5% | 1.6% | 0.63 | 🟡 **Moderately under-attended** |
| NUMBER | 2.8% | 2.8% | 1.02 | 🟢 **Proportional** |
| OTHER | 40.0% | 25.0% | 0.63 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
