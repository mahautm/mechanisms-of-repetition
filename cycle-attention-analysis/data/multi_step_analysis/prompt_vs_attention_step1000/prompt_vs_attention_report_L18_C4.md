# Prompt vs Attention Analysis Report
**Layer 18, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 14.3% | 2.14 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.50 | 🔵 **Under-attended** |
| SENTENCE_END | 3.5% | 3.7% | 1.05 | 🟢 **Proportional** |
| PUNCTUATION | 3.3% | 1.1% | 0.33 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 54.7% | 1.02 | 🟢 **Proportional** |
| BRACKET | 0.9% | 1.0% | 1.09 | 🟢 **Proportional** |
| NUMBER | 1.8% | 0.7% | 0.38 | 🔵 **Under-attended** |
| OTHER | 30.0% | 24.3% | 0.81 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 8.7% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.2% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 4.0% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 57.2% | 0.57 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 9.4% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 20.5% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER
