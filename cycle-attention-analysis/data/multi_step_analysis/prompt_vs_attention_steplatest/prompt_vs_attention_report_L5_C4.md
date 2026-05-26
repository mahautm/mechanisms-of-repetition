# Prompt vs Attention Analysis Report
**Layer 5, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 3.1% | 85.1% | 27.24 | 🔴 **Over-attended** |
| SENTENCE_END | 13.1% | 2.6% | 0.20 | 🔵 **Under-attended** |
| PUNCTUATION | 3.1% | 0.7% | 0.21 | 🔵 **Under-attended** |
| CONTENT_WORD | 78.1% | 11.7% | 0.15 | 🔵 **Under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 2.5% | 0.0% | 0.00 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: OTHER
- **High attention categories** (>20%): TEMPLATE_WORD

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
