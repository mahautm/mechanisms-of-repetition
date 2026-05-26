# Prompt vs Attention Analysis Report
**Layer 2, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 12.2% | 1.82 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 1.3% | 2.67 | 🔴 **Over-attended** |
| SENTENCE_END | 3.5% | 7.9% | 2.25 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 2.8% | 0.84 | 🟢 **Proportional** |
| CONTENT_WORD | 53.4% | 46.9% | 0.88 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.1% | 0.16 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.5% | 0.26 | 🔵 **Under-attended** |
| OTHER | 30.0% | 28.3% | 0.94 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 17.7% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 6.7% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 58.7% | 0.59 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 0.0% | 16.9% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD
