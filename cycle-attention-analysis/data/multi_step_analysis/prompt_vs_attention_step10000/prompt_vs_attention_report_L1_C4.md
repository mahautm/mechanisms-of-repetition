# Prompt vs Attention Analysis Report
**Layer 1, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 8.9% | 1.29 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 1.2% | 2.32 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 10.1% | 2.95 | 🔴 **Over-attended** |
| PUNCTUATION | 3.2% | 4.4% | 1.35 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 56.2% | 1.05 | 🟢 **Proportional** |
| BRACKET | 0.8% | 0.1% | 0.12 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 2.0% | 1.15 | 🟢 **Proportional** |
| OTHER | 29.9% | 17.1% | 0.57 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 14.8% | 1.10 | 🟢 **Proportional** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 9.4% | 1.06 | 🟢 **Proportional** |
| PUNCTUATION | 5.0% | 5.7% | 1.15 | 🟢 **Proportional** |
| CONTENT_WORD | 43.0% | 47.1% | 1.10 | 🟢 **Proportional** |
| BRACKET | 1.3% | 1.5% | 1.12 | 🟢 **Proportional** |
| NUMBER | 3.6% | 0.5% | 0.15 | 🔵 **Under-attended** |
| OTHER | 24.8% | 21.0% | 0.85 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: PUNCTUATION
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER
