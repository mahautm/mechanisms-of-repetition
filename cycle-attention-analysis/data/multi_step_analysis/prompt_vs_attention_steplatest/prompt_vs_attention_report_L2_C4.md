# Prompt vs Attention Analysis Report
**Layer 2, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 20.8% | 3.14 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 3.2% | 6.32 | 🔴 **Over-attended** |
| SENTENCE_END | 3.6% | 9.3% | 2.61 | 🔴 **Over-attended** |
| PUNCTUATION | 3.5% | 2.5% | 0.72 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 54.8% | 44.5% | 0.81 | 🟢 **Proportional** |
| BRACKET | 0.7% | 0.5% | 0.68 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 1.6% | 0.89 | 🟢 **Proportional** |
| OTHER | 28.6% | 17.7% | 0.62 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: OTHER
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 17.5% | 1.70 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 0.97 | 🟢 **Proportional** |
| SENTENCE_END | 5.5% | 9.1% | 1.65 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.0% | 3.2% | 1.58 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 36.5% | 29.7% | 0.81 | 🟢 **Proportional** |
| BRACKET | 2.5% | 5.2% | 2.07 | 🔴 **Over-attended** |
| NUMBER | 2.8% | 0.8% | 0.30 | 🔵 **Under-attended** |
| OTHER | 40.0% | 34.2% | 0.85 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: BRACKET
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER
