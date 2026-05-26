# Prompt vs Attention Analysis Report
**Layer 20, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 87.7% | 13.24 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.55 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.6% | 0.6% | 0.17 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.2% | 0.05 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 4.1% | 0.08 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.2% | 0.24 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.3% | 0.20 | 🔵 **Under-attended** |
| OTHER | 28.6% | 6.6% | 0.23 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 40.7% | 3.97 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.05 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.4% | 0.81 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.1% | 0.06 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 31.4% | 0.86 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.3% | 0.10 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.2% | 0.80 | 🟡 **Moderately under-attended** |
| OTHER | 40.0% | 20.9% | 0.52 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
