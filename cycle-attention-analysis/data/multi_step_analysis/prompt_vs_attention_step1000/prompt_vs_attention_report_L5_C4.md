# Prompt vs Attention Analysis Report
**Layer 5, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 8.2% | 1.23 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 1.0% | 1.97 | 🟡 **Moderately over-attended** |
| SENTENCE_END | 3.5% | 6.0% | 1.70 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 2.7% | 0.81 | 🟢 **Proportional** |
| CONTENT_WORD | 53.4% | 58.9% | 1.10 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.0% | 0.04 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.8% | 0.44 | 🔵 **Under-attended** |
| OTHER | 30.0% | 22.4% | 0.75 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 4.7% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 4.1% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 55.0% | 0.55 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 36.0% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER
