# Prompt vs Attention Analysis Report
**Layer 16, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 85.2% | 12.86 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 1.02 | 🟢 **Proportional** |
| SENTENCE_END | 3.6% | 0.0% | 0.01 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.0% | 0.01 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 2.7% | 0.05 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.3% | 0.41 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.6% | 0.33 | 🔵 **Under-attended** |
| OTHER | 28.6% | 10.7% | 0.37 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 40.3% | 3.93 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.8% | 0.87 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 30.7% | 0.84 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.2% | 0.07 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.3% | 0.84 | 🟢 **Proportional** |
| OTHER | 40.0% | 21.7% | 0.54 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
