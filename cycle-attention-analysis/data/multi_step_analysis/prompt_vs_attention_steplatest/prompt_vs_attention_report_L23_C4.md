# Prompt vs Attention Analysis Report
**Layer 23, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 40.5% | 6.11 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 2.7% | 5.35 | 🔴 **Over-attended** |
| SENTENCE_END | 3.6% | 2.0% | 0.55 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.5% | 0.4% | 0.12 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 12.2% | 0.22 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.9% | 1.24 | 🟡 **Moderately over-attended** |
| NUMBER | 1.8% | 1.9% | 1.06 | 🟢 **Proportional** |
| OTHER | 28.6% | 39.5% | 1.38 | 🟡 **Moderately over-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 24.1% | 2.35 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.41 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 6.3% | 1.14 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.2% | 0.10 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 36.5% | 1.00 | 🟢 **Proportional** |
| BRACKET | 2.5% | 1.2% | 0.47 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.0% | 0.74 | 🟡 **Moderately under-attended** |
| OTHER | 40.0% | 29.5% | 0.74 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
