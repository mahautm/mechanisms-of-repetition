# Prompt vs Attention Analysis Report
**Layer 16, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 84.9% | 12.35 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 0.96 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 0.1% | 0.04 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.4% | 0.11 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 3.6% | 0.07 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.5% | 0.66 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.5% | 0.29 | 🔵 **Under-attended** |
| OTHER | 29.6% | 9.4% | 0.32 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: SENTENCE_END
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 30.2% | 5.56 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.2% | 1.41 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.4% | 0.87 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 34.7% | 0.67 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.4% | 0.23 | 🔵 **Under-attended** |
| OTHER | 29.8% | 23.0% | 0.77 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
