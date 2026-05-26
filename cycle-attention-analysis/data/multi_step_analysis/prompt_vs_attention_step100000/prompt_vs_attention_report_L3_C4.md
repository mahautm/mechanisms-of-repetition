# Prompt vs Attention Analysis Report
**Layer 3, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.9% | 25.3% | 3.68 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.5% | 1.0% | 2.07 | 🔴 **Over-attended** |
| SENTENCE_END | 3.4% | 5.6% | 1.64 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 3.3% | 2.5% | 0.75 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 53.8% | 50.0% | 0.93 | 🟢 **Proportional** |
| BRACKET | 0.8% | 0.2% | 0.20 | 🔵 **Under-attended** |
| NUMBER | 1.8% | 1.5% | 0.85 | 🟢 **Proportional** |
| OTHER | 29.6% | 14.0% | 0.47 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 5.4% | 20.5% | 3.77 | 🔴 **Over-attended** |
| TEMPLATE_WORD | 0.3% | 0.4% | 1.56 | 🟡 **Moderately over-attended** |
| SENTENCE_END | 6.6% | 7.9% | 1.20 | 🟡 **Moderately over-attended** |
| PUNCTUATION | 2.7% | 3.0% | 1.12 | 🟢 **Proportional** |
| CONTENT_WORD | 51.9% | 42.9% | 0.83 | 🟢 **Proportional** |
| BRACKET | 1.6% | 1.4% | 0.84 | 🟢 **Proportional** |
| NUMBER | 1.6% | 0.7% | 0.44 | 🔵 **Under-attended** |
| OTHER | 29.8% | 23.1% | 0.77 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: NUMBER
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD, OTHER
