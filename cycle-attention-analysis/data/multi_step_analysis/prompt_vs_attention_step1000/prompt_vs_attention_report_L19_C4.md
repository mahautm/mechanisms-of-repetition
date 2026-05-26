# Prompt vs Attention Analysis Report
**Layer 19, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 16.7% | 2.50 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.54 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 2.0% | 0.56 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 0.5% | 0.16 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 57.6% | 1.08 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.4% | 0.23 | 🔵 **Under-attended** |
| OTHER | 30.0% | 22.6% | 0.75 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 2.0% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 62.5% | 0.62 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.9% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 34.6% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER
