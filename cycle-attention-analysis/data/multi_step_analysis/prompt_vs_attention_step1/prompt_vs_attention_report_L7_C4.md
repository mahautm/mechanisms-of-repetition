# Prompt vs Attention Analysis Report
**Layer 7, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 3.1% | 2.6% | 0.84 | 🟢 **Proportional** |
| SENTENCE_END | 13.1% | 2.4% | 0.19 | 🔵 **Under-attended** |
| PUNCTUATION | 3.1% | 0.2% | 0.05 | 🔵 **Under-attended** |
| CONTENT_WORD | 78.1% | 67.1% | 0.86 | 🟢 **Proportional** |
| BRACKET | 0.0% | 1.3% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 1.6% | ∞ | 🔴 **Over-attended** |
| OTHER | 2.5% | 24.7% | 9.87 | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: NUMBER
- **Most under-attended**: NEWLINE
- **High attention categories** (>20%): CONTENT_WORD, OTHER

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
