# Prompt vs Attention Analysis Report
**Layer 18, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 1.0% | 0.15 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.06 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 0.8% | 0.22 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 1.0% | 0.30 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 62.4% | 1.17 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.4% | 0.43 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.6% | 0.89 | 🟢 **Proportional** |
| OTHER | 30.0% | 32.9% | 1.10 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: CONTENT_WORD
- **Most under-attended**: TEMPLATE_WORD
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
