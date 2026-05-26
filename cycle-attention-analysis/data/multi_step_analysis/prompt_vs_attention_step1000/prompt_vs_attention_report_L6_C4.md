# Prompt vs Attention Analysis Report
**Layer 6, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 10.1% | 1.51 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.22 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 4.7% | 1.35 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 8.9% | 2.68 | 🔴 **Over-attended** |
| CONTENT_WORD | 53.4% | 51.3% | 0.96 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.6% | 0.73 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 0.9% | 0.51 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 23.3% | 0.78 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: PUNCTUATION
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.7% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 70.0% | 0.70 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.5% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 0.0% | 28.7% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER
