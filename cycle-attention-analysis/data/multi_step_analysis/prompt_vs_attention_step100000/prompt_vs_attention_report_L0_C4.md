# Prompt vs Attention Analysis Report
**Layer 0, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 7.0% | 1.01 | 🟢 **Proportional** |
| TEMPLATE_WORD | 0.5% | 0.7% | 1.47 | 🟡 **Moderately over-attended** |
| SENTENCE_END | 3.4% | 14.5% | 4.23 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 3.7% | 1.11 | 🟢 **Proportional** |
| CONTENT_WORD | 53.8% | 55.6% | 1.03 | 🟢 **Proportional** |
| BRACKET | 0.8% | 1.2% | 1.45 | 🟡 **Moderately over-attended** |
| NUMBER | 1.8% | 1.0% | 0.57 | 🟡 **Moderately under-attended** |
| OTHER | 29.6% | 16.3% | 0.55 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: OTHER
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 9.6% | 1.76 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.3% | 0.9% | 3.30 | 🔴 **Over-attended** |
| SENTENCE_END | 6.6% | 9.3% | 1.41 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 4.7% | 1.72 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 51.9% | 45.9% | 0.88 | 🟢 **Proportional** |
| BRACKET | 1.6% | 1.7% | 1.02 | 🟢 **Proportional** |
| NUMBER | 1.6% | 0.7% | 0.40 | 🔵 **Under-attended** |
| OTHER | 29.8% | 27.4% | 0.92 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER
