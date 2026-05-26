# Prompt vs Attention Analysis Report
**Layer 11, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 8.4% | 1.25 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.63 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 11.7% | 3.34 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 4.9% | 1.48 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 50.1% | 0.94 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.4% | 0.48 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.2% | 0.68 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 23.0% | 0.77 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: SENTENCE_END
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 1.1% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.7% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 6.0% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 72.7% | 0.73 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 0.0% | 19.5% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD
