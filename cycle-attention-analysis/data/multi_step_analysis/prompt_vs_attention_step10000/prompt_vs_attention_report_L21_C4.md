# Prompt vs Attention Analysis Report
**Layer 21, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 84.4% | 12.16 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 0.2% | 0.46 | 🔵 **Under-attended** |
| SENTENCE_END | 3.4% | 1.6% | 0.47 | 🔵 **Under-attended** |
| PUNCTUATION | 3.2% | 0.3% | 0.09 | 🔵 **Under-attended** |
| CONTENT_WORD | 53.4% | 7.8% | 0.15 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.2% | 0.26 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 0.4% | 0.20 | 🔵 **Under-attended** |
| OTHER | 29.9% | 5.1% | 0.17 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: PUNCTUATION
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 13.5% | 35.0% | 2.60 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.0% | 0.0% | 0.00 | 🔵 **Under-attended** |
| SENTENCE_END | 8.8% | 12.3% | 1.39 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 5.0% | 1.2% | 0.24 | 🔵 **Under-attended** |
| CONTENT_WORD | 43.0% | 32.9% | 0.76 | 🟡 **Moderately under-attended** |
| BRACKET | 1.3% | 0.1% | 0.11 | 🔵 **Under-attended** |
| NUMBER | 3.6% | 4.4% | 1.22 | 🟡 **Moderately over-attended** |
| OTHER | 24.8% | 14.1% | 0.57 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: TEMPLATE_WORD
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
