# Cross-Step Attention Bias Analysis

## Processing Summary

### step1
- ❌ **Status**: Failed or skipped
- **Error**: Bias ratio summary failed

### step1000
- ✅ **Status**: Successfully processed
- 📊 **Reports**: plots/multi_step_analysis/prompt_vs_attention_step1000
- 📈 **Summary**: plots/multi_step_analysis/bias_summary_step1000

### step10000
- ✅ **Status**: Successfully processed
- 📊 **Reports**: plots/multi_step_analysis/prompt_vs_attention_step10000
- 📈 **Summary**: plots/multi_step_analysis/bias_summary_step10000

### step100000
- ✅ **Status**: Successfully processed
- 📊 **Reports**: plots/multi_step_analysis/prompt_vs_attention_step100000
- 📈 **Summary**: plots/multi_step_analysis/bias_summary_step100000

### steplatest
- ✅ **Status**: Successfully processed
- 📊 **Reports**: plots/multi_step_analysis/prompt_vs_attention_steplatest
- 📈 **Summary**: plots/multi_step_analysis/bias_summary_steplatest

## Analysis Overview

Each step directory contains:
1. **Individual layer reports**: Detailed bias ratios for each layer
2. **Summary visualizations**: Heatmaps, bar charts, and evolution plots
3. **Statistical summaries**: Key findings and interpretations

## Next Steps

To compare across training steps:
1. Compare the `attention_bias_summary.png` files across steps
2. Look at how newline vs content word bias changes over training
3. Examine if template word specialization emerges gradually