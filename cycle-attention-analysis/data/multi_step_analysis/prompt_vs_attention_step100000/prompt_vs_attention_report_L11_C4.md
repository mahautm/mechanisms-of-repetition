# Prompt vs Attention Analysis Report
**Layer 11, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 72.2% | 10.50 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 0.97 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 1.8% | 0.53 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 0.8% | 0.26 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 13.6% | 0.25 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.5% | 0.65 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.9% | 0.50 | 🟡 **Moderately under-attended** |
| OTHER | 29.6% | 9.7% | 0.33 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: CONTENT_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 24.6% | 4.53 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.2% | 0.64 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 6.6% | 9.8% | 1.49 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.4% | 0.89 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 38.0% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.2% | 0.13 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.4% | 0.23 | 🔵 **Under-attended** |
| OTHER | 29.8% | 24.4% | 0.82 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
