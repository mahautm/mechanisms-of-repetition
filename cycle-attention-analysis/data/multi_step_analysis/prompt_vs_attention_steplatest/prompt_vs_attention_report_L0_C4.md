# Prompt vs Attention Analysis Report
**Layer 0, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 9.5% | 1.44 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 1.7% | 3.47 | 🔴 **Over-attended** |
| SENTENCE_END | 3.6% | 10.8% | 3.02 | 🔴 **Over-attended** |
| PUNCTUATION | 3.5% | 3.4% | 0.98 | 🟢 **Proportional** |
| CONTENT_WORD | 54.8% | 56.2% | 1.03 | 🟢 **Proportional** |
| BRACKET | 0.7% | 0.4% | 0.62 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 1.1% | 0.63 | 🟡 **Moderately under-attended** |
| OTHER | 28.6% | 16.8% | 0.59 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: OTHER
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 13.1% | 1.27 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.76 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 5.5% | 10.1% | 1.83 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.0% | 2.4% | 1.18 | 🟢 **Proportional** |
| CONTENT_WORD | 36.5% | 30.3% | 0.83 | 🟢 **Proportional** |
| BRACKET | 2.5% | 4.2% | 1.67 | 🟡 **Moderately over-attended** |
| NUMBER | 2.8% | 1.0% | 0.38 | 🔵 **Under-attended** |
| OTHER | 40.0% | 38.6% | 0.96 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER
