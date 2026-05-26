# Prompt vs Attention Analysis Report
**Layer 9, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 11.7% | 1.76 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.6% | 1.21 | 🟡 **Moderately over-attended** |
| SENTENCE_END | 3.5% | 9.9% | 2.82 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 4.5% | 1.37 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 54.4% | 1.02 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.1% | 0.15 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.9% | 0.53 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 17.8% | 0.59 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 4.7% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 1.9% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 76.4% | 0.76 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 0.3% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 16.5% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD
