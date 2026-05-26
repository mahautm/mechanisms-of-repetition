# Prompt vs Attention Analysis Report
**Layer 14, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 85.7% | 12.94 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.81 | 🟢 **Proportional** |
| SENTENCE_END | 3.6% | 0.5% | 0.14 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.3% | 0.07 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 6.5% | 0.12 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.0% | 0.06 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.6% | 0.37 | 🔵 **Under-attended** |
| OTHER | 28.6% | 6.0% | 0.21 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 39.0% | 3.80 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.24 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.7% | 0.85 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.1% | 0.04 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 31.1% | 0.85 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.2% | 0.08 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.4% | 0.87 | 🟢 **Proportional** |
| OTHER | 40.0% | 22.4% | 0.56 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
