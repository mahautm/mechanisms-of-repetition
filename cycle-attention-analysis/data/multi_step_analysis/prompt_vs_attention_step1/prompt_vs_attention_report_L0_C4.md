# Prompt vs Attention Analysis Report
**Layer 0, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 0.7% | 0.10 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.59 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 1.0% | 0.29 | 🔵 **Under-attended** |
| PUNCTUATION | 3.3% | 1.1% | 0.33 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 60.8% | 1.14 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.4% | 0.49 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 2.1% | 1.17 | 🟢 **Proportional** |
| OTHER | 30.0% | 33.7% | 1.12 | 🟢 **Proportional** |

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
