# Prompt vs Attention Analysis Report
**Layer 10, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 70.0% | 10.18 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.56 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.4% | 1.9% | 0.55 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 1.2% | 0.37 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.8% | 16.4% | 0.30 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.6% | 0.71 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.5% | 0.29 | 🔵 **Under-attended** |
| OTHER | 29.6% | 9.2% | 0.31 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 23.6% | 4.34 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.1% | 0.49 | 🔵 **Under-attended** |
| SENTENCE_END | 6.6% | 9.2% | 1.40 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 2.1% | 0.79 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 51.9% | 37.1% | 0.71 | 🟡 **Moderately under-attended** |
| BRACKET | 1.6% | 0.3% | 0.15 | 🔵 **Under-attended** |
| NUMBER | 1.6% | 0.3% | 0.20 | 🔵 **Under-attended** |
| OTHER | 29.8% | 27.3% | 0.91 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
