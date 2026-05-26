# Prompt vs Attention Analysis Report
**Layer 3, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 31.9% | 4.81 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 1.3% | 2.58 | 🔴 **Over-attended** |
| SENTENCE_END | 3.6% | 3.9% | 1.09 | 🟢 **Proportional** |
| PUNCTUATION | 3.5% | 2.3% | 0.65 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 54.8% | 46.8% | 0.85 | 🟢 **Proportional** |
| BRACKET | 0.7% | 0.3% | 0.47 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.2% | 0.66 | 🟡 **Moderately under-attended** |
| OTHER | 28.6% | 12.4% | 0.43 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: OTHER
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 23.6% | 2.30 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.69 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 5.5% | 5.4% | 0.99 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 1.8% | 0.88 | 🟢 **Proportional** |
| CONTENT_WORD | 36.5% | 35.0% | 0.96 | 🟢 **Proportional** |
| BRACKET | 2.5% | 2.9% | 1.14 | 🟢 **Proportional** |
| NUMBER | 2.8% | 1.0% | 0.37 | 🔵 **Under-attended** |
| OTHER | 40.0% | 30.0% | 0.75 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
