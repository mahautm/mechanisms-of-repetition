# Prompt vs Attention Analysis Report
**Layer 6, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 71.5% | 10.40 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.16 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 3.9% | 1.12 | 🟢 **Proportional** |
| PUNCTUATION | 3.3% | 1.5% | 0.46 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 15.4% | 0.29 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.6% | 0.73 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 1.0% | 0.58 | 🟡 **Moderately under-attended** |
| OTHER | 29.6% | 6.0% | 0.20 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 22.5% | 4.13 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.2% | 0.57 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 6.6% | 8.9% | 1.36 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 3.3% | 1.23 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 51.9% | 38.1% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.9% | 0.58 | 🟡 **Moderately under-attended** |
| NUMBER | 1.6% | 0.1% | 0.07 | 🔵 **Under-attended** |
| OTHER | 29.8% | 26.0% | 0.87 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
