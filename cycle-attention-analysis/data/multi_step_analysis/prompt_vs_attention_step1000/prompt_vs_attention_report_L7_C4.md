# Prompt vs Attention Analysis Report
**Layer 7, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 13.7% | 2.05 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.50 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 6.1% | 1.74 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 5.6% | 1.69 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 52.6% | 0.99 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.4% | 0.46 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.7% | 0.40 | 🔵 **Under-attended** |
| OTHER | 30.0% | 20.7% | 0.69 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 1.7% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 83.0% | 0.83 | 🟢 **Proportional** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.9% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 14.3% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD
