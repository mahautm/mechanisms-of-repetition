# Prompt vs Attention Analysis Report
**Layer 15, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 89.5% | 13.50 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.37 | 🔵 **Under-attended** |
| SENTENCE_END | 3.6% | 0.2% | 0.05 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.2% | 0.07 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 6.2% | 0.11 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.2% | 0.10 | 🔵 **Under-attended** |
| OTHER | 28.6% | 3.5% | 0.12 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 40.1% | 3.91 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.05 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.7% | 0.85 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.1% | 0.03 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 31.3% | 0.86 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.2% | 0.08 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.4% | 0.86 | 🟢 **Proportional** |
| OTHER | 40.0% | 21.3% | 0.53 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
