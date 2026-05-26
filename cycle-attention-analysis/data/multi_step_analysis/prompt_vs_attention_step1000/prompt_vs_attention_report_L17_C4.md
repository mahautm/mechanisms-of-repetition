# Prompt vs Attention Analysis Report
**Layer 17, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 20.8% | 3.12 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.3% | 0.66 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 10.8% | 3.07 | 🔴 **Over-attended** |
| PUNCTUATION | 3.3% | 2.0% | 0.61 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 44.0% | 0.83 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.4% | 0.47 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.5% | 0.27 | 🔵 **Under-attended** |
| OTHER | 30.0% | 21.1% | 0.70 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 2.3% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 51.4% | 0.51 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.7% | ∞ | 🔴 **Over-attended** |
| NUMBER | 0.0% | 9.8% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 35.8% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD, OTHER
