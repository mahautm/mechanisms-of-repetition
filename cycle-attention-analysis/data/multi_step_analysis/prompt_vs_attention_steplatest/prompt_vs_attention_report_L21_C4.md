# Prompt vs Attention Analysis Report
**Layer 21, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 78.7% | 11.87 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.7% | 1.41 | 🟡 **Moderately over-attended** |
| SENTENCE_END | 3.6% | 2.2% | 0.61 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 3.5% | 0.3% | 0.08 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 7.2% | 0.13 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.3% | 0.39 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.7% | 0.42 | 🔵 **Under-attended** |
| OTHER | 28.6% | 10.0% | 0.35 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 38.3% | 3.73 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.29 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.7% | 0.85 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.4% | 0.18 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 31.8% | 0.87 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.4% | 0.18 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.2% | 0.81 | 🟢 **Proportional** |
| OTHER | 40.0% | 22.0% | 0.55 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
