# Prompt vs Attention Analysis Report
**Layer 12, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 13.9% | 2.08 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.7% | 1.45 | 🟡 **Moderately over-attended** |
| SENTENCE_END | 3.5% | 2.8% | 0.79 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 2.0% | 0.60 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 58.5% | 1.10 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.3% | 0.33 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.2% | 0.70 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 20.5% | 0.68 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 1.0% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.6% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 92.8% | 0.93 | 🟢 **Proportional** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.7% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 4.9% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD
