# Prompt vs Attention Analysis Report
**Layer 4, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 52.3% | 7.90 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 1.2% | 2.42 | 🔴 **Over-attended** |
| SENTENCE_END | 3.6% | 1.7% | 0.47 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 1.0% | 0.30 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 22.1% | 0.40 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.6% | 0.82 | 🟢 **Proportional** |
| NUMBER | 1.8% | 1.6% | 0.91 | 🟢 **Proportional** |
| OTHER | 28.6% | 19.5% | 0.68 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 27.1% | 2.65 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 5.2% | 0.95 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.8% | 0.38 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 35.3% | 0.97 | 🟢 **Proportional** |
| BRACKET | 2.5% | 1.3% | 0.53 | 🟡 **Moderately under-attended** |
| NUMBER | 2.8% | 2.5% | 0.90 | 🟢 **Proportional** |
| OTHER | 40.0% | 27.8% | 0.70 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
