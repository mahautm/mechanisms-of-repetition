# Prompt vs Attention Analysis Report
**Layer 17, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 92.1% | 13.40 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.23 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 0.5% | 0.14 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.2% | 0.05 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 4.7% | 0.09 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.19 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.0% | 0.02 | 🔵 **Under-attended** |
| OTHER | 29.6% | 2.3% | 0.08 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 31.7% | 5.84 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 8.5% | 1.30 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.2% | 0.82 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 34.4% | 0.66 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.2% | 0.11 | 🔵 **Under-attended** |
| OTHER | 29.8% | 22.9% | 0.77 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
