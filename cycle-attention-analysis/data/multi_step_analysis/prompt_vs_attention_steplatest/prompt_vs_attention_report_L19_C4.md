# Prompt vs Attention Analysis Report
**Layer 19, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 87.3% | 13.17 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.4% | 0.79 | 🟡 **Moderately under-attended** |
| SENTENCE_END | 3.6% | 0.4% | 0.12 | 🔵 **Under-attended** |
| PUNCTUATION | 3.5% | 0.2% | 0.05 | 🔵 **Under-attended** |
| CONTENT_WORD | 54.8% | 6.8% | 0.12 | 🔵 **Under-attended** |
| BRACKET | 0.7% | 0.2% | 0.26 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.3% | 0.18 | 🔵 **Under-attended** |
| OTHER | 28.6% | 4.4% | 0.15 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.2% | 40.7% | 3.97 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.1% | 0.18 | 🔵 **Under-attended** |
| SENTENCE_END | 5.5% | 4.7% | 0.85 | 🟢 **Proportional** |
| PUNCTUATION | 2.0% | 0.2% | 0.10 | 🔵 **Under-attended** |
| CONTENT_WORD | 36.5% | 31.2% | 0.85 | 🟢 **Proportional** |
| BRACKET | 2.5% | 0.1% | 0.03 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 2.2% | 0.82 | 🟢 **Proportional** |
| OTHER | 40.0% | 20.9% | 0.52 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
