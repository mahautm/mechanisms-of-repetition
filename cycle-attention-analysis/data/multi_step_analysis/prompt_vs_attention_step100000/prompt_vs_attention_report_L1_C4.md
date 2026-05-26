# Prompt vs Attention Analysis Report
**Layer 1, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 11.0% | 1.60 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.6% | 1.20 | 🟢 **Proportional** |
| SENTENCE_END | 3.4% | 15.9% | 4.62 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 4.1% | 1.25 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.8% | 54.2% | 1.01 | 🟢 **Proportional** |
| BRACKET | 0.8% | 1.4% | 1.71 | 🟡 **Moderately over-attended** |
| NUMBER | 1.8% | 0.9% | 0.50 | 🔵 **Under-attended** |
| OTHER | 29.6% | 11.9% | 0.40 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: OTHER
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 10.9% | 2.01 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.6% | 2.30 | 🔴 **Over-attended** |
| SENTENCE_END | 6.6% | 6.3% | 0.97 | 🟢 **Proportional** |
| PUNCTUATION | 2.7% | 3.5% | 1.29 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 51.9% | 52.9% | 1.02 | 🟢 **Proportional** |
| BRACKET | 1.6% | 2.1% | 1.27 | 🟡 **Moderately over-attended** |
| NUMBER | 1.6% | 0.9% | 0.52 | 🟡 **Moderately under-attended** |
| OTHER | 29.8% | 22.7% | 0.76 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER
