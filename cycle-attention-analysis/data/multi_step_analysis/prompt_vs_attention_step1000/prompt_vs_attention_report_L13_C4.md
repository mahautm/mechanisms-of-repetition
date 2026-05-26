# Prompt vs Attention Analysis Report
**Layer 13, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 12.7% | 1.90 | 🟡 **Moderately over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.86 | 🟢 **Proportional** |
| SENTENCE_END | 3.5% | 2.6% | 0.76 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 2.4% | 0.71 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 51.6% | 0.97 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.0% | 0.05 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.0% | 0.58 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 29.2% | 0.97 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 13.3% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 3.0% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 61.2% | 0.61 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 3.0% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 19.4% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD
