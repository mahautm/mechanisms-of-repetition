# Prompt vs Attention Analysis Report
**Layer 2, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 11.5% | 1.65 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 5.7% | 11.34 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 5.7% | 1.66 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.2% | 3.9% | 1.21 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 50.1% | 0.94 | 🟢 **Proportional** |
| BRACKET | 0.8% | 0.1% | 0.12 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.1% | 0.65 | 🟡 **Moderately under-attended** |
| OTHER | 29.9% | 21.9% | 0.73 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 12.5% | 0.93 | 🟢 **Proportional** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 12.6% | 1.43 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 6.2% | 1.25 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 43.0% | 30.4% | 0.71 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 2.1% | 1.55 | 🟡 **Moderately over-attended** |
| NUMBER | 3.6% | 0.9% | 0.26 | 🔵 **Under-attended** |
| OTHER | 24.8% | 35.2% | 1.42 | 🟡 **Moderately over-attended** |

### Key Findings:
- **Most over-attended**: BRACKET
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER
