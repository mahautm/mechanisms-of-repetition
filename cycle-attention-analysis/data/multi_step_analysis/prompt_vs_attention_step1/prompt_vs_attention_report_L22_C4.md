# Prompt vs Attention Analysis Report
**Layer 22, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 2.2% | 0.32 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.48 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 1.1% | 0.33 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 0.7% | 0.21 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 59.3% | 1.11 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.3% | 0.33 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.3% | 0.75 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 34.8% | 1.16 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: PUNCTUATION
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
