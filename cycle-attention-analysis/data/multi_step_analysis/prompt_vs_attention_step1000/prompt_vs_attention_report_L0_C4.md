# Prompt vs Attention Analysis Report
**Layer 0, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 12.2% | 1.82 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.39 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 20.0% | 5.73 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 4.8% | 1.43 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 38.6% | 0.72 | 🟡 **Moderately under-attended** |
| BRACKET | 0.9% | 0.7% | 0.75 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.3% | 0.14 | 🔵 **Under-attended** |
| OTHER | 30.0% | 23.3% | 0.78 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): SENTENCE_END, CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.4% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 3.6% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 75.7% | 0.76 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 2.6% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 6.9% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 10.8% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD
