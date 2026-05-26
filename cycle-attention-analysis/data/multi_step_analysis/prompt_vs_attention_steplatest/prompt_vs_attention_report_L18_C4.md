# Prompt vs Attention Analysis Report
**Layer 18, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 85.1% | 12.85 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.6% | 1.14 | 🟢 **Proportional** |
| SENTENCE_END | 3.6% | 0.0% | 0.01 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.0% | 0.01 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 1.4% | 0.03 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.3% | 0.47 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.7% | 0.39 | 🔵 **Under-attended** |
| OTHER | 28.6% | 11.8% | 0.41 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 38.9% | 3.80 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 5.3% | 0.96 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 31.9% | 0.87 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.3% | 0.10 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.2% | 0.81 | 🟢 **Proportional** |
| OTHER | 40.0% | 21.4% | 0.54 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
