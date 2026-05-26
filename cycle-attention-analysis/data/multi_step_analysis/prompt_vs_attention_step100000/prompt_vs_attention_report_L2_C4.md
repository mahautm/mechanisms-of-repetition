# Prompt vs Attention Analysis Report
**Layer 2, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 19.4% | 2.82 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 2.1% | 4.21 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 10.3% | 3.00 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 3.3% | 0.99 | 🟢 **Proportional** |
| CONTENT_WORD | 53.8% | 44.1% | 0.82 | 🟢 **Proportional** |
| BRACKET | 0.8% | 0.8% | 0.92 | 🟢 **Proportional** |
| NUMBER | 1.8% | 1.1% | 0.64 | 🟡 **Moderately under-attended** |
| OTHER | 29.6% | 18.9% | 0.64 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 16.5% | 3.04 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.6% | 2.24 | 🔴 **Over-attended** |
| SENTENCE_END | 6.6% | 7.2% | 1.10 | 🟢 **Proportional** |
| PUNCTUATION | 2.7% | 3.9% | 1.42 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 51.9% | 37.8% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 1.9% | 1.16 | 🟢 **Proportional** |
| NUMBER | 1.6% | 1.0% | 0.58 | 🟡 **Moderately under-attended** |
| OTHER | 29.8% | 31.2% | 1.05 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER
