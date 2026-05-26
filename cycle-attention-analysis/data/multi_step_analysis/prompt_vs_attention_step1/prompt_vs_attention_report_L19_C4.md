# Prompt vs Attention Analysis Report
**Layer 19, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 1.4% | 0.21 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.63 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 0.9% | 0.26 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 1.1% | 0.34 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 61.2% | 1.15 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.3% | 0.34 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.5% | 0.88 | 🟢 **Proportional** |
| OTHER | 30.0% | 33.3% | 1.11 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: CONTENT_WORD
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
