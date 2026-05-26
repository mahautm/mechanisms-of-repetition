# Prompt vs Attention Analysis Report
**Layer 14, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 11.3% | 1.69 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.62 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 6.0% | 1.71 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 2.2% | 0.66 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 56.3% | 1.06 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.2% | 0.28 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.0% | 0.55 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 22.7% | 0.76 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 7.0% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 2.4% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 75.8% | 0.76 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.4% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 0.2% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 14.2% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD
