# Prompt vs Attention Analysis Report
**Layer 1, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 15.2% | 2.21 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 1.0% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 7.5% | 18.5% | 2.47 | 🔴 **Over-attended** |
| PUNCTUATION | 3.1% | 5.9% | 1.88 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 35.6% | 39.6% | 1.11 | 🟢 **Proportional** |
| BRACKET | 0.6% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.2% | 3.1% | 2.47 | 🔴 **Over-attended** |
| OTHER | 45.0% | 16.8% | 0.37 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: OTHER
