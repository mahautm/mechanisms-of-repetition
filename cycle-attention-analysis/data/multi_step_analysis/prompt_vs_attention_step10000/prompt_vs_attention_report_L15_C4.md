# Prompt vs Attention Analysis Report
**Layer 15, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 88.6% | 12.77 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.32 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 1.2% | 0.35 | 🔵 **Under-attended** |
| PUNCTUATION | 3.2% | 0.0% | 0.02 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 5.5% | 0.10 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.1% | 0.08 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 29.9% | 4.4% | 0.15 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 36.5% | 2.71 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 15.7% | 1.77 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 0.5% | 0.10 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 32.1% | 0.75 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 3.9% | 1.10 | 🟢 **Proportional** |
| OTHER | 24.8% | 11.3% | 0.45 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
