# Prompt vs Attention Analysis Report
**Layer 8, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 3.1% | 2.7% | 0.87 | 🟢 **Proportional** |
| SENTENCE_END | 13.1% | 2.3% | 0.17 | 🔵 **Under-attended** |
| PUNCTUATION | 3.1% | 0.9% | 0.30 | 🔵 **Under-attended** |
| CONTENT_WORD | 78.1% | 71.4% | 0.91 | 🟢 **Proportional** |
| BRACKET | 0.0% | 0.4% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 1.9% | ∞ | 🔴 **Over-attended** |
| OTHER | 2.5% | 20.3% | 8.13 | 🔴 **Over-attended** |

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
