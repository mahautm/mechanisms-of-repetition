# Prompt vs Attention Analysis Report
**Layer 22, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 81.5% | 11.86 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.74 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.4% | 1.7% | 0.50 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 0.4% | 0.11 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 8.0% | 0.15 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.28 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.4% | 0.22 | 🔵 **Under-attended** |
| OTHER | 29.6% | 7.4% | 0.25 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 28.1% | 5.17 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.1% | 0.23 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 8.7% | 1.32 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.4% | 0.88 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 36.0% | 0.69 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.1% | 0.03 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.5% | 0.30 | 🔵 **Under-attended** |
| OTHER | 29.8% | 24.3% | 0.81 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
