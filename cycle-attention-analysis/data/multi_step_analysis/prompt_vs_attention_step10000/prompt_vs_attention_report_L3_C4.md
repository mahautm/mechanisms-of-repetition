# Prompt vs Attention Analysis Report
**Layer 3, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 7.3% | 1.06 | 🟢 **Proportional** |
| TEMPLATE_WORD | 0.5% | 2.1% | 4.30 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 6.4% | 1.85 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.2% | 3.6% | 1.09 | 🟢 **Proportional** |
| CONTENT_WORD | 53.4% | 60.1% | 1.12 | 🟢 **Proportional** |
| BRACKET | 0.8% | 0.1% | 0.08 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 2.4% | 1.39 | 🟡 **Moderately over-attended** |
| OTHER | 29.9% | 18.0% | 0.60 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 10.5% | 0.78 | 🟡 **Moderately under-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 11.4% | 1.29 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 3.3% | 0.66 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 43.0% | 45.4% | 1.06 | 🟢 **Proportional** |
| BRACKET | 1.3% | 1.4% | 1.08 | 🟢 **Proportional** |
| NUMBER | 3.6% | 2.3% | 0.66 | 🟡 **Moderately under-attended** |
| OTHER | 24.8% | 25.6% | 1.03 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER
