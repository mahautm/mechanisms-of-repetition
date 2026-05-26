# Prompt vs Attention Analysis Report
**Layer 17, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 0.6% | 0.08 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.07 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 1.0% | 0.30 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.7% | 0.21 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 63.6% | 1.19 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.1% | 0.16 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 2.5% | 1.45 | 🟡 **Moderately over-attended** |
| OTHER | 30.0% | 31.4% | 1.05 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NUMBER
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
