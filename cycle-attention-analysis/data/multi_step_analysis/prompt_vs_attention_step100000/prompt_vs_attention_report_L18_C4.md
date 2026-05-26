# Prompt vs Attention Analysis Report
**Layer 18, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 82.4% | 11.99 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.6% | 1.19 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 0.1% | 0.04 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.1% | 0.03 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 2.2% | 0.04 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.7% | 0.85 | 🟢 **Proportional** |
| NUMBER | 1.8% | 0.9% | 0.50 | 🟡 **Moderately under-attended** |
| OTHER | 29.6% | 13.0% | 0.44 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 29.2% | 5.37 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.2% | 1.40 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.4% | 0.90 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 34.7% | 0.67 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.1% | 0.05 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.4% | 0.27 | 🔵 **Under-attended** |
| OTHER | 29.8% | 23.9% | 0.80 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
