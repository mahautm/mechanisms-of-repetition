# Prompt vs Attention Analysis Report
**Layer 18, 4 Cycles**

## Natural Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 6.6% | 78.7% | 11.87 | 🔴 **Over-attended** |
| SENTENCE_END | 3.6% | 2.2% | 0.61 | 🟡 **Moderately under-attended** |
| PUNCTUATION | 7.2% | 0.9% | 0.12 | 🔵 **Under-attended** |
| CONTENT_WORD | 48.6% | 7.9% | 0.16 | 🔵 **Under-attended** |
| FUNCTION_WORD | 22.3% | 5.2% | 0.23 | 🔵 **Under-attended** |
| PROGRAMMING | 2.7% | 0.8% | 0.31 | 🔵 **Under-attended** |
| BRACKET | 0.8% | 0.0% | 0.05 | 🔵 **Under-attended** |
| NUMBER | 2.8% | 0.9% | 0.31 | 🔵 **Under-attended** |
| OTHER | 5.5% | 3.4% | 0.63 | 🟡 **Moderately under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE

## No Cycle Icl Sequences

| Token Type | Prompt % | Attention % | Bias Ratio | Interpretation |
|------------|----------|-------------|------------|----------------|
| NEWLINE | 10.5% | 40.3% | 3.84 | 🔴 **Over-attended** |
| SENTENCE_END | 5.5% | 4.7% | 0.85 | 🟢 **Proportional** |
| PUNCTUATION | 7.8% | 5.2% | 0.67 | 🟡 **Moderately under-attended** |
| CONTENT_WORD | 34.3% | 31.4% | 0.91 | 🟢 **Proportional** |
| FUNCTION_WORD | 18.2% | 7.7% | 0.42 | 🔵 **Under-attended** |
| PROGRAMMING | 4.5% | 2.7% | 0.60 | 🟡 **Moderately under-attended** |
| BRACKET | 1.5% | 0.2% | 0.14 | 🔵 **Under-attended** |
| NUMBER | 4.2% | 4.3% | 1.01 | 🟢 **Proportional** |
| OTHER | 13.5% | 3.4% | 0.26 | 🔵 **Under-attended** |

### Key Findings:
- **Most over-attended**: NEWLINE
- **Most under-attended**: BRACKET
- **High attention categories** (>20%): NEWLINE, CONTENT_WORD
