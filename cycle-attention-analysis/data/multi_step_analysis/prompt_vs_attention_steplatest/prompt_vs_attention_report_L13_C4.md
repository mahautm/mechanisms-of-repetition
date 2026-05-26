# Prompt vs Attention Analysis Report
**Layer 13, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 85.8% | 12.95 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 1.02 | 🟢 **Proportional** |
| SENTENCE_END | 3.6% | 0.3% | 0.09 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.1% | 0.02 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 5.8% | 0.11 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.2% | 0.23 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.5% | 0.30 | 🔵 **Under-attended** |
| OTHER | 28.6% | 6.8% | 0.24 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 40.6% | 3.97 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.14 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.4% | 0.80 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.1% | 0.03 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 31.1% | 0.85 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.3% | 0.10 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.5% | 0.91 | 🟢 **Proportional** |
| OTHER | 40.0% | 21.0% | 0.52 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
