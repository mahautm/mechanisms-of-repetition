# Prompt vs Attention Analysis Report
**Layer 12, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 75.7% | 11.02 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.39 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 1.3% | 0.38 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.7% | 0.22 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 15.0% | 0.28 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.3% | 0.43 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.8% | 0.45 | 🔵 **Under-attended** |
| OTHER | 29.6% | 5.9% | 0.20 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: OTHER
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 25.1% | 4.62 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.1% | 0.43 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.1% | 1.38 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.9% | 1.08 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 38.1% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.2% | 0.15 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.1% | 0.08 | 🔵 **Under-attended** |
| OTHER | 29.8% | 24.3% | 0.81 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
