# Prompt vs Attention Analysis Report
**Layer 3, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 3.3% | 0.50 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.5% | 1.05 | 🟢 **Proportional** |
| SENTENCE_END | 3.5% | 5.8% | 1.66 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 4.6% | 1.38 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 62.6% | 1.17 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.0% | 0.01 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.6% | 0.37 | 🔵 **Under-attended** |
| OTHER | 30.0% | 22.5% | 0.75 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.2% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 2.2% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 74.7% | 0.75 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 0.0% | 22.8% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER
