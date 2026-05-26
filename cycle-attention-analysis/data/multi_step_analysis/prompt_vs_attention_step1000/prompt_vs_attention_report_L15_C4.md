# Prompt vs Attention Analysis Report
**Layer 15, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 18.7% | 2.80 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.70 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.5% | 3.2% | 0.93 | 🟢 **Proportional** |
| PUNCTUATION | 3.3% | 3.0% | 0.91 | 🟢 **Proportional** |
| CONTENT_WORD | 53.4% | 48.9% | 0.92 | 🟢 **Proportional** |
| BRACKET | 0.9% | 0.4% | 0.42 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.6% | 0.32 | 🔵 **Under-attended** |
| OTHER | 30.0% | 24.9% | 0.83 | 🟢 **Proportional** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD, OTHER

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 5.3% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.2% | ∞ | 🔴 **Over-attended** |
| SENTENCE_END | 0.0% | 2.0% | ∞ | 🔴 **Over-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 70.3% | 0.70 | 🟡 **Moderately under-attended** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 1.4% | ∞ | 🔴 **Over-attended** |
| OTHER | 0.0% | 20.8% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): CONTENT_WORD, OTHER
