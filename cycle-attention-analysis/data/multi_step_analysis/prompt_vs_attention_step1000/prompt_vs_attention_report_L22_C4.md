# Prompt vs Attention Analysis Report
**Layer 22, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 11.9% | 1.78 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.17 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 6.7% | 1.91 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 2.6% | 0.79 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 57.4% | 1.07 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.5% | 0.53 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.7% | 0.40 | 🔵 **Under-attended** |
| OTHER | 30.0% | 20.2% | 0.67 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.5% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.8% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 83.7% | 0.84 | 🟢 **Proportional** |
| BRACKET | 0.0% | 0.8% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 3.4% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 10.8% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD
