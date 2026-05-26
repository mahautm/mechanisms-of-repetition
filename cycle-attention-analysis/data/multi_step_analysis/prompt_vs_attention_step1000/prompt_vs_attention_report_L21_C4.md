# Prompt vs Attention Analysis Report
**Layer 21, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 21.2% | 3.17 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.02 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 8.8% | 2.51 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 4.9% | 1.46 | 🟡 **Moderately over-attended** |
| CONTENT_WORD | 53.4% | 45.5% | 0.85 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.5% | 0.54 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 1.0% | 0.54 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 18.3% | 0.61 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 18.8% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 65.1% | 0.65 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 3.7% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 8.2% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 4.2% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD
