# Prompt vs Attention Analysis Report
**Layer 23, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 1.9% | 0.29 | 🔵 **Under-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.56 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 2.5% | 0.72 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.3% | 2.9% | 0.88 | 🟢 **Proportional** |
| CONTENT_WORD | 53.4% | 71.7% | 1.34 | 🟡 **Moderately over-attended** |
| BRACKET | 0.9% | 0.2% | 0.18 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.4% | 0.25 | 🔵 **Under-attended** |
| OTHER | 30.0% | 20.1% | 0.67 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: CONTENT_WORD
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 1.4% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.1% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.3% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 96.4% | 0.96 | 🟢 **Proportional** |
| BRACKET | 0.0% | 0.7% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 0.2% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 0.9% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD
