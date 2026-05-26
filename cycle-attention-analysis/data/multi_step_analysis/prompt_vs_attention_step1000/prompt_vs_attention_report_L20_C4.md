# Prompt vs Attention Analysis Report
**Layer 20, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 28.1% | 4.21 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.0% | 0.10 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 6.3% | 1.81 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 2.2% | 0.66 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 45.3% | 0.85 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.6% | 0.66 | 🟡 **Moderately under-attended** |
| NUMBER | 1.8% | 1.3% | 0.73 | 🟡 **Moderately under-attended** |
| OTHER | 30.0% | 16.2% | 0.54 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 12.7% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.4% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 59.5% | 0.60 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.9% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 6.0% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 20.5% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD, OTHER
