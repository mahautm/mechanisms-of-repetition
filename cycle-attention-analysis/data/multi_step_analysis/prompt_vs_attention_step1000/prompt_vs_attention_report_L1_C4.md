# Prompt vs Attention Analysis Report
**Layer 1, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.7% | 4.5% | 0.68 | 🟡 **Moderately under-attended** |
| TEMPLATE_WORD | 0.5% | 0.9% | 1.84 | 🟡 **Moderately over-attended** |
| SENTENCE_END | 3.5% | 3.5% | 1.00 | 🟢 **Proportional** |
| PUNCTUATION | 3.3% | 2.0% | 0.60 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.4% | 71.9% | 1.35 | 🟡 **Moderately over-attended** |
| BRACKET | 0.9% | 0.3% | 0.33 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.3% | 0.17 | 🔵 **Under-attended** |
| OTHER | 30.0% | 16.6% | 0.55 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: TEMPLATE_WORD
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 0.0% | 3.3% | ∞ | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| PUNCTUATION | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| CONTENT_WORD | 100.0% | 92.6% | 0.93 | 🟢 **Proportional** |
| BRACKET | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| NUMBER | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| OTHER | 0.0% | 4.2% | ∞ | 🔴 **Over-attended** |

### Key Findings:
- **Most over-attended**: OTHER
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): CONTENT_WORD
