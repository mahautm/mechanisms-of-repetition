from pathlib import Path
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors
import tempfile
import shutil


def cleanup_log_files(base_path, model_name="EleutherAI/pythia-1.4b", checkpoints=None, dry_run=False):
    """
    Clean up log files by removing lines that begin with 'batch'
    
    Args:
        base_path: Base output directory containing multihead analysis results
        model_name: Model name (default: "EleutherAI/pythia-1.4b")
        checkpoints: List of checkpoint names to clean (default: all available)
        dry_run: If True, only report what would be cleaned without making changes
        
    Returns:
        dict: Summary of cleanup operations performed
    """
    model_path = Path(base_path) / model_name.replace("/", "/")
    
    if checkpoints is None:
        # Find all available checkpoints
        if model_path.exists():
            checkpoints = [d.name for d in model_path.iterdir() if d.is_dir()]
        else:
            print(f"Model path not found: {model_path}")
            return {}
    
    cleanup_summary = {
        'files_processed': 0,
        'lines_removed': 0,
        'files_with_changes': 0,
        'errors': []
    }
    
    for checkpoint in checkpoints:
        checkpoint_path = model_path / checkpoint
        
        if not checkpoint_path.exists():
            print(f"Checkpoint path not found: {checkpoint_path}")
            continue
        
        # Find all log files in this checkpoint
        log_files = list(checkpoint_path.rglob("*.out"))
        
        for log_file in log_files:
            try:
                cleanup_summary['files_processed'] += 1
                
                # Read the file
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                # Filter out lines starting with 'batch'
                original_count = len(lines)
                filtered_lines = [line for line in lines if not line.strip().startswith('batch')]
                lines_removed = original_count - len(filtered_lines)
                
                if lines_removed > 0:
                    cleanup_summary['lines_removed'] += lines_removed
                    cleanup_summary['files_with_changes'] += 1
                    
                    if dry_run:
                        print(f"[DRY RUN] Would remove {lines_removed} batch lines from {log_file}")
                    else:
                        # Write back to file using a temporary file for safety
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                                       dir=log_file.parent, 
                                                       prefix=f"{log_file.name}.tmp") as tmp_file:
                            tmp_file.writelines(filtered_lines)
                            temp_path = tmp_file.name
                        
                        # Replace original file with cleaned version
                        shutil.move(temp_path, log_file)
                        print(f"Cleaned {lines_removed} batch lines from {log_file}")
                
            except Exception as e:
                error_msg = f"Error processing {log_file}: {str(e)}"
                cleanup_summary['errors'].append(error_msg)
                print(error_msg)
    
    # Print summary
    print(f"\nCleanup Summary:")
    print(f"Files processed: {cleanup_summary['files_processed']}")
    print(f"Files with changes: {cleanup_summary['files_with_changes']}")
    print(f"Total lines removed: {cleanup_summary['lines_removed']}")
    if cleanup_summary['errors']:
        print(f"Errors encountered: {len(cleanup_summary['errors'])}")
    
    return cleanup_summary


def load_repetition_evolution_from_outputs(base_path, model_name="EleutherAI/pythia-1.4b", 
                                          checkpoints=None, layer=20, cycles=None, max_length=32):
    """
    Load repetition evolution data by parsing output files from multihead analysis
    
    Args:
        base_path: Base output directory containing multihead analysis results
        model_name: Model name (default: "EleutherAI/pythia-1.4b")
        checkpoints: List of checkpoint names to load (default: all available)
        layer: Target layer number
        cycles: List of cycle numbers to load (default: all available)
        max_length: Maximum sequence length used in analysis
        
    Returns:
        Dictionary with checkpoint keys containing repetition data
    """
    import re
    
    model_path = Path(base_path) / model_name.replace("/", "/")
    
    if checkpoints is None:
        # Find all available checkpoints
        if model_path.exists():
            checkpoints = [d.name for d in model_path.iterdir() if d.is_dir()]
        else:
            print(f"Model path not found: {model_path}")
            return {}
    
    if cycles is None:
        cycles = [0, 1, 2]  # Default cycles
    
    repetition_data = {}
    
    for checkpoint in checkpoints:
        checkpoint_path = model_path / checkpoint / f"layer_{layer}"
        
        if not checkpoint_path.exists():
            print(f"Layer path not found: {checkpoint_path}")
            continue
        
        # Try to find output files for any available cycle
        found_data = False
        for cycle in cycles:
            output_file = checkpoint_path / f"full_analysis_cyc{cycle}_ml{max_length}.out"
            
            if output_file.exists():
                try:
                    # Parse the output file
                    with open(output_file, 'r') as f:
                        content = f.read()
                    
                    # Extract data using regex patterns
                    data_index_match = re.search(rf'layer {layer} data index: \[(.*?)\]', content)
                    repetition_index_match = re.search(rf'layer {layer} repetition index: \[(.*?)\]', content)
                    
                    if data_index_match and repetition_index_match:
                        # Parse the indices
                        data_indices_str = data_index_match.group(1)
                        repetition_indices_str = repetition_index_match.group(1)
                        
                        # Convert to lists of integers
                        data_indices = [int(x.strip()) for x in data_indices_str.split(',') if x.strip()]
                        repetition_indices = [int(x.strip()) for x in repetition_indices_str.split(',') if x.strip()]
                        
                        repetition_data[checkpoint] = {
                            'data_indices': data_indices,
                            'repetition_indices': repetition_indices,
                            'cycle': cycle,
                            'layer': layer
                        }
                        
                        found_data = True
                        print(f"Loaded {checkpoint} layer {layer} cycle {cycle}: {len(data_indices)} datapoints, {len(repetition_indices)} repetitions")
                        break  # Use first available cycle
                        
                except Exception as e:
                    print(f"Error parsing {output_file}: {e}")
                    continue
        
        if not found_data:
            print(f"No valid data found for {checkpoint} layer {layer}")
    
    return repetition_data


def load_multihead_results_across_cycles(base_path, model_name="EleutherAI/pythia-1.4b", checkpoints=None, 
                                        cycle_range=None, max_length=32):
    """
    Load multi-head analysis results across both checkpoints and cycles
    
    Args:
        base_path: Base output directory
        model_name: Model name
        checkpoints: List of checkpoints to analyze
        cycle_range: List of cycle numbers to analyze (e.g., [0, 1, 2, 3, 4, 5])
        max_length: Maximum sequence length used in analysis
        
    Returns:
        dict: Contains results organized by checkpoint -> cycle -> layer -> head_values
    """
    if checkpoints is None:
        checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    
    if cycle_range is None:
        cycle_range = [0, 1, 2, 3, 4, 5]
    
    base_dir = Path(base_path) / model_name
    
    results = {
        'natural': {},  # checkpoint -> cycle -> layer -> head_values
        'icl': {},
        'successful_icl': {},  # checkpoint -> cycle -> layer -> head_values for successful ICL
        'no_cycle_icl': {},    # checkpoint -> cycle -> layer -> head_values for no-cycle ICL
        'metadata': {
            'model_name': model_name,
            'cycle_range': cycle_range,
            'max_length': max_length,
            'checkpoints': checkpoints
        }
    }
    
    for checkpoint in checkpoints:
        checkpoint_dir = base_dir / checkpoint
        if not checkpoint_dir.exists():
            print(f"Warning: Checkpoint directory {checkpoint_dir} does not exist")
            continue
            
        print(f"Loading results for checkpoint: {checkpoint}")
        
        # Initialize checkpoint results
        results['natural'][checkpoint] = {}
        results['icl'][checkpoint] = {}
        results['successful_icl'][checkpoint] = {}
        results['no_cycle_icl'][checkpoint] = {}
        
        for n_cycles in cycle_range:
            results['natural'][checkpoint][n_cycles] = {}
            results['icl'][checkpoint][n_cycles] = {}
            results['successful_icl'][checkpoint][n_cycles] = {}
            results['no_cycle_icl'][checkpoint][n_cycles] = {}
            
            # Process each layer for this cycle count
            for layer_dir in sorted(checkpoint_dir.glob("layer_*")):
                layer_idx = int(layer_dir.name.split("_")[1])
                
                # Find the output log file for this cycle count
                log_file = layer_dir / f"full_analysis_cyc{n_cycles}_ml{max_length}.out"
                
                if not log_file.exists():
                    # Try alternative naming patterns
                    alt_patterns = [
                        layer_dir / f"full_analysis_cycles{n_cycles}_ml{max_length}.out",
                        layer_dir / f"analysis_cyc{n_cycles}_ml{max_length}.out",
                        layer_dir / f"full_analysis_cyc{n_cycles}.out"
                    ]
                    
                    found = False
                    for alt_log in alt_patterns:
                        if alt_log.exists():
                            log_file = alt_log
                            found = True
                            break
                    
                    if not found:
                        continue
                
                # Parse the log file (now returns 4 values)
                natural_heatmap, icl_heatmap, successful_icl_heatmap, no_cycle_icl_heatmap = parse_multihead_log(log_file, layer_idx)
                
                if natural_heatmap is not None:
                    results['natural'][checkpoint][n_cycles][layer_idx] = natural_heatmap
                if icl_heatmap is not None:
                    results['icl'][checkpoint][n_cycles][layer_idx] = icl_heatmap
                if successful_icl_heatmap is not None:
                    results['successful_icl'][checkpoint][n_cycles][layer_idx] = successful_icl_heatmap
                if no_cycle_icl_heatmap is not None:
                    results['no_cycle_icl'][checkpoint][n_cycles][layer_idx] = no_cycle_icl_heatmap
                    
        print(f"Loaded cycles {cycle_range} for checkpoint {checkpoint}")
    
    return results


def create_cycle_evolution_dataframe(results_across_cycles, heatmap_type='natural'):
    """
    Convert multi-head results across cycles into a pandas DataFrame
    
    Args:
        results_across_cycles: Results dictionary from load_multihead_results_across_cycles
        heatmap_type: 'natural', 'icl', or 'successful_icl'
        
    Returns:
        pd.DataFrame: With columns ['checkpoint', 'cycle', 'layer', 'head', 'contrast', 'layer_head']
    """
    data_rows = []
    
    for checkpoint, cycles in results_across_cycles[heatmap_type].items():
        for cycle, layers in cycles.items():
            for layer_idx, head_values in layers.items():
                if head_values is not None:
                    for head_idx, contrast_value in enumerate(head_values):
                        data_rows.append({
                            'checkpoint': checkpoint,
                            'cycle': cycle,
                            'layer': layer_idx,
                            'head': head_idx,
                            'contrast': float(contrast_value),
                            'layer_head': f"{layer_idx}.{head_idx}",
                            'category': heatmap_type
                        })
    
    return pd.DataFrame(data_rows)


def plot_cycle_evolution_by_checkpoint(results_across_cycles, save_path=None, max_heads_to_show=8):
    """
    Create horizontal plots showing ICL contrast evolution across cycles for each checkpoint,
    skipping step1 if empty, and adding natural steplatest at the end
    
    Args:
        results_across_cycles: Results dictionary from load_multihead_results_across_cycles
        save_path: Path to save the plot (optional)
        max_heads_to_show: Maximum number of heads to highlight per plot
    """
    # Create DataFrames
    natural_df = create_cycle_evolution_dataframe(results_across_cycles, 'natural')
    icl_df = create_cycle_evolution_dataframe(results_across_cycles, 'icl')

    # natural_df, icl_df = icl_df, natural_df  # Swap to have ICL first


    
    if icl_df.empty:
        print("No ICL data available for plotting cycle evolution")
        return
    
    # Custom sorting function for checkpoints
    def sort_checkpoint_key(checkpoint):
        if checkpoint == 'steplatest':
            return float('inf')  # Put steplatest at the end
        elif checkpoint.startswith('step'):
            try:
                return int(checkpoint[4:])  # Extract number after 'step'
            except ValueError:
                return 0
        else:
            return 0
    
    # Filter ICL checkpoints: skip step1, step5000, and step7000, keep others, sort by step number
    icl_checkpoints = []
    for checkpoint in sorted(icl_df['checkpoint'].unique(), key=sort_checkpoint_key):
        checkpoint_data = icl_df[icl_df['checkpoint'] == checkpoint]
        # Skip step1 if it has no meaningful data or very few datapoints
        if checkpoint == 'step1':
            if len(checkpoint_data) < 10:  # Skip if too little data
                print(f"Skipping {checkpoint} - insufficient ICL data ({len(checkpoint_data)} datapoints)")
                continue
        # Skip step5000 and step7000 to save space in the figure
        if checkpoint in ['step5000', 'step7000']:
            print(f"Skipping {checkpoint} - excluded to save space")
            continue
        icl_checkpoints.append(checkpoint)
    
    # Add natural steplatest at the end if available
    add_natural_steplatest = False
    if not natural_df.empty and 'steplatest' in natural_df['checkpoint'].unique():
        add_natural_steplatest = True
        natural_steplatest = natural_df[natural_df['checkpoint'] == 'steplatest']
    
    total_plots = len(icl_checkpoints) + int(add_natural_steplatest)
    
    if total_plots == 0:
        print("No valid checkpoints for plotting")
        return
    
    # Find interesting heads globally to ensure consistent colors
    all_icl_data = icl_df[icl_df['checkpoint'].isin(icl_checkpoints)]
    global_head_variances = all_icl_data.groupby('layer_head')['contrast'].var().sort_values(ascending=False)
    global_interesting_heads = global_head_variances.head(max_heads_to_show).index.tolist()
    
    # Create consistent color and style mapping for all heads
    head_color_map = {}
    head_style_map = {}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    for i, head in enumerate(global_interesting_heads):
        head_color_map[head] = colors[i % len(colors)]
        head_style_map[head] = styles[i % len(styles)]
    
    # Calculate shared y-axis limits for ALL plots (ICL + natural)
    all_contrast_values = []
    all_contrast_values.extend(icl_df['contrast'].values)
    if add_natural_steplatest:
        all_contrast_values.extend(natural_steplatest['contrast'].values)
    
    if len(all_contrast_values) > 0:
        y_min = min(all_contrast_values)
        y_max = max(all_contrast_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        y_min_padded = y_min - y_padding
        y_max_padded = y_max + y_padding
    else:
        y_min_padded, y_max_padded = 0, 1
    
    # Create horizontal subplot layout optimized for A4 paper
    # Smaller figure size makes text appear larger relative to plot
    base_width = 3.0 * len(icl_checkpoints)  # Smaller base width per plot for better text visibility
    fig_width = base_width + (3.0 if add_natural_steplatest else 0)  # Compact spacing
    
    if add_natural_steplatest and len(icl_checkpoints) > 0:
        # Use gridspec for custom spacing - smaller gap for A4 layout
        import matplotlib.gridspec as gridspec
        
        # Create width ratios: equal for ICL plots, small gap, then natural plot
        width_ratios = [1] * len(icl_checkpoints) + [0.1, 1]  # Keep small gap, equal natural plot
        
        fig = plt.figure(figsize=(fig_width, 3.0))  # Smaller height for paper readability
        gs = gridspec.GridSpec(1, len(icl_checkpoints) + 2, width_ratios=width_ratios, 
                              wspace=0.2)  # Slightly more spacing for clarity
        
        # Create ICL subplot axes
        axes = []
        for i in range(len(icl_checkpoints)):
            axes.append(fig.add_subplot(gs[0, i]))
        
        # Skip the gap column (gs[0, len(icl_checkpoints)]) 
        # Add natural plot axis
        axes.append(fig.add_subplot(gs[0, len(icl_checkpoints) + 1]))
    else:
        fig, axes = plt.subplots(1, total_plots, figsize=(fig_width, 3.0))
    
    # Ensure axes is always a list
    if total_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot ICL contrasts for each checkpoint
    for checkpoint in icl_checkpoints:
        checkpoint_icl = icl_df[icl_df['checkpoint'] == checkpoint]
        
        if not checkpoint_icl.empty:
            # Plot all heads in light grey background
            sns.lineplot(data=checkpoint_icl, x="cycle", y="contrast", hue="layer_head",
                        alpha=0.1, legend=False, ax=axes[plot_idx], color="lightgrey")
            
            # Highlight interesting heads with consistent colors and styles
            interesting_df = checkpoint_icl[checkpoint_icl['layer_head'].isin(global_interesting_heads)]
            if not interesting_df.empty:
                for head in global_interesting_heads:
                    head_data = interesting_df[interesting_df['layer_head'] == head]
                    if not head_data.empty:
                        axes[plot_idx].plot(head_data['cycle'], head_data['contrast'],
                                          color=head_color_map[head], linestyle=head_style_map[head],
                                          marker='o', markersize=6, linewidth=2, label=head)
            
            # Use simplified checkpoint label (remove 'step' prefix, use K notation)
            checkpoint_label = checkpoint.replace('step', '')
            if checkpoint_label.isdigit():
                num = int(checkpoint_label)
                if num >= 1000:
                    checkpoint_label = f"{num // 1000}K"
            axes[plot_idx].set_title(checkpoint_label, fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel("Cycle Number", fontsize=12)
            # Only show y-axis label on the first plot
            if plot_idx == 0:
                axes[plot_idx].set_ylabel("Contrast", fontsize=12)
            else:
                axes[plot_idx].set_ylabel("")
            axes[plot_idx].tick_params(axis='both', which='major', labelsize=10)  # Larger tick labels for A4
            axes[plot_idx].legend().set_visible(False)  # Hide legend for cleaner look
            axes[plot_idx].set_ylim(y_min_padded, y_max_padded)  # Same scale for all
            axes[plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Add natural steplatest plot at the end with visual separation
    if add_natural_steplatest:
        # Find heads with highest variance for natural data but use global color mapping
        head_variances = natural_steplatest.groupby('layer_head')['contrast'].var().sort_values(ascending=False)
        natural_interesting_heads = head_variances.head(max_heads_to_show).index.tolist()
        
        # Extend color mapping for natural-specific heads if needed
        for head in natural_interesting_heads:
            if head not in head_color_map:
                next_color_idx = len(head_color_map) % len(colors)
                next_style_idx = len(head_color_map) % len(styles)
                head_color_map[head] = colors[next_color_idx]
                head_style_map[head] = styles[next_style_idx]
        
        # Add visual separation: subtle background color and clear title
        axes[plot_idx].set_facecolor('#f8f9fa')  # Very light grey background
        axes[plot_idx].set_title('Natural Data', fontsize=16, fontweight='bold', 
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Add a subtle border to emphasize separation
        for spine in axes[plot_idx].spines.values():
            spine.set_linewidth(2)
            spine.set_color('navy')
        
        # Plot all heads in light grey background
        sns.lineplot(data=natural_steplatest, x="cycle", y="contrast", hue="layer_head",
                    alpha=0.1, legend=False, ax=axes[plot_idx], color="lightgrey")
        
        # Highlight interesting heads with consistent colors and styles
        interesting_df = natural_steplatest[natural_steplatest['layer_head'].isin(natural_interesting_heads)]
        if not interesting_df.empty:
            for head in natural_interesting_heads:
                head_data = interesting_df[interesting_df['layer_head'] == head]
                if not head_data.empty:
                    axes[plot_idx].plot(head_data['cycle'], head_data['contrast'],
                                      color=head_color_map[head], linestyle=head_style_map[head],
                                      marker='o', markersize=6, linewidth=2, label=head)
        
        axes[plot_idx].set_title("latest", fontsize=14, fontweight='bold')
        axes[plot_idx].set_xlabel("Cycle Number", fontsize=12)
        # Natural plot doesn't need y-axis label since it's not the first plot
        axes[plot_idx].tick_params(axis='both', which='major', labelsize=10)  # Larger tick labels for A4
        axes[plot_idx].legend().set_visible(False)
        axes[plot_idx].set_ylim(y_min_padded, y_max_padded)  # Same scale as ICL plots
        axes[plot_idx].grid(True, alpha=0.3)
    
    # Add section labels above the plots, properly centered using actual axes positions
    # Need to draw the figure first to get axes positions
    plt.tight_layout()
    
    if len(icl_checkpoints) > 0:
        # Calculate center of ICL section using actual axes positions
        icl_left = axes[0].get_position().x0
        icl_right = axes[len(icl_checkpoints)-1].get_position().x1
        icl_center = (icl_left + icl_right) / 2.0
        fig.text(icl_center, 0.99, 'ICL', fontsize=16, fontweight='bold', ha='center', transform=fig.transFigure)
    
    if add_natural_steplatest:
        # Natural plot is the last axis
        natural_ax = axes[-1]
        natural_left = natural_ax.get_position().x0
        natural_right = natural_ax.get_position().x1
        natural_center = (natural_left + natural_right) / 2.0
        fig.text(natural_center, 0.99, 'Natural', fontsize=16, fontweight='bold', ha='center', transform=fig.transFigure)
    
    # Create a single legend for all plots showing the head mappings
    if global_interesting_heads:
        legend_elements = []
        all_heads_shown = set(global_interesting_heads)
        if add_natural_steplatest:
            all_heads_shown.update(natural_interesting_heads)
        
        for head in sorted(all_heads_shown):
            legend_elements.append(plt.Line2D([0], [0], color=head_color_map[head], 
                                            linestyle=head_style_map[head], marker='o',
                                            markersize=6, linewidth=2, label=head))
        
        # Position legend below the plots in 2 rows for better layout
        # Calculate number of columns to split legend into 2 rows
        ncol = (len(legend_elements) + 1) // 2  # Ceiling division to split into 2 rows
        fig.legend(handles=legend_elements, loc='lower center', ncol=ncol, 
                  bbox_to_anchor=(0.5, -0.3), fontsize=11)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved horizontal cycle evolution plot to {save_path}")
    
    plt.show()


def plot_cycle_summary_statistics(results_across_cycles, save_path=None):
    """
    Create summary plots showing aggregate statistics across cycles and checkpoints
    
    Args:
        results_across_cycles: Results dictionary from load_multihead_results_across_cycles
        save_path: Path to save the plot (optional)
    """
    # Create DataFrames
    natural_df = create_cycle_evolution_dataframe(results_across_cycles, 'natural')
    icl_df = create_cycle_evolution_dataframe(results_across_cycles, 'icl')
    
    if natural_df.empty and icl_df.empty:
        print("No data available for summary statistics")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean contrast by cycle and checkpoint (Natural)
    if not natural_df.empty:
        summary_natural = natural_df.groupby(['checkpoint', 'cycle'])['contrast'].agg(['mean', 'std']).reset_index()
        
        for checkpoint in summary_natural['checkpoint'].unique():
            checkpoint_data = summary_natural[summary_natural['checkpoint'] == checkpoint]
            axes[0, 0].plot(checkpoint_data['cycle'], checkpoint_data['mean'], 
                          marker='o', label=checkpoint, linewidth=2)
            axes[0, 0].fill_between(checkpoint_data['cycle'], 
                                  checkpoint_data['mean'] - checkpoint_data['std'],
                                  checkpoint_data['mean'] + checkpoint_data['std'],
                                  alpha=0.2)
        
        axes[0, 0].set_title("Mean Natural Contrast by Cycle", fontsize=12)
        axes[0, 0].set_xlabel("Cycle Number")
        axes[0, 0].set_ylabel("Mean Contrast")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean contrast by cycle and checkpoint (ICL)
    if not icl_df.empty:
        summary_icl = icl_df.groupby(['checkpoint', 'cycle'])['contrast'].agg(['mean', 'std']).reset_index()
        
        for checkpoint in summary_icl['checkpoint'].unique():
            checkpoint_data = summary_icl[summary_icl['checkpoint'] == checkpoint]
            axes[0, 1].plot(checkpoint_data['cycle'], checkpoint_data['mean'], 
                          marker='o', label=checkpoint, linewidth=2)
            axes[0, 1].fill_between(checkpoint_data['cycle'], 
                                  checkpoint_data['mean'] - checkpoint_data['std'],
                                  checkpoint_data['mean'] + checkpoint_data['std'],
                                  alpha=0.2)
        
        axes[0, 1].set_title("Mean ICL Contrast by Cycle", fontsize=12)
        axes[0, 1].set_xlabel("Cycle Number")
        axes[0, 1].set_ylabel("Mean Contrast")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Contrast variance by cycle (Natural)
    if not natural_df.empty:
        variance_natural = natural_df.groupby(['checkpoint', 'cycle'])['contrast'].var().reset_index()
        
        for checkpoint in variance_natural['checkpoint'].unique():
            checkpoint_data = variance_natural[variance_natural['checkpoint'] == checkpoint]
            axes[1, 0].plot(checkpoint_data['cycle'], checkpoint_data['contrast'], 
                          marker='s', label=checkpoint, linewidth=2)
        
        axes[1, 0].set_title("Natural Contrast Variance by Cycle", fontsize=12)
        axes[1, 0].set_xlabel("Cycle Number")
        axes[1, 0].set_ylabel("Contrast Variance")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Contrast variance by cycle (ICL)
    if not icl_df.empty:
        variance_icl = icl_df.groupby(['checkpoint', 'cycle'])['contrast'].var().reset_index()
        
        for checkpoint in variance_icl['checkpoint'].unique():
            checkpoint_data = variance_icl[variance_icl['checkpoint'] == checkpoint]
            axes[1, 1].plot(checkpoint_data['cycle'], checkpoint_data['contrast'], 
                          marker='s', label=checkpoint, linewidth=2)
        
        axes[1, 1].set_title("ICL Contrast Variance by Cycle", fontsize=12)
        axes[1, 1].set_xlabel("Cycle Number")
        axes[1, 1].set_ylabel("Contrast Variance")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved cycle summary statistics to {save_path}")
    
    plt.show()


def load_multihead_results(base_path, model_name="EleutherAI/pythia-1.4b", checkpoints=None, n_cycles=5, max_length=32):
    """
    Load multi-head analysis results from the directory structure created by run_full_multihead_analysis.sh
    
    Args:
        base_path: Base output directory (e.g., "/home/mmahaut/projects/parrots/outputs_multihead_full")
        model_name: Model name (e.g., "EleutherAI/pythia-1.4b")
        checkpoints: List of checkpoints to analyze (e.g., ["step1", "step1000", "steplatest"])
        n_cycles: Number of cycles to analyze
        max_length: Maximum sequence length used in analysis
        
    Returns:
        dict: Contains 'natural', 'icl', and 'successful_icl' heatmaps organized by checkpoint and layer
    """
    if checkpoints is None:
        checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    
    base_dir = Path(base_path) / model_name
    
    results = {
        'natural': {},  # checkpoint -> layer -> head_values
        'icl': {},
        'successful_icl': {},  # checkpoint -> layer -> head_values for successful ICL predictions
        'no_cycle_icl': {},    # checkpoint -> layer -> head_values for no-cycle ICL
        'metadata': {
            'model_name': model_name,
            'n_cycles': n_cycles,
            'max_length': max_length,
            'checkpoints': checkpoints
        }
    }
    
    for checkpoint in checkpoints:
        checkpoint_dir = base_dir / checkpoint
        if not checkpoint_dir.exists():
            print(f"Warning: Checkpoint directory {checkpoint_dir} does not exist")
            continue
            
        print(f"Loading results for checkpoint: {checkpoint}")
        
        # Initialize checkpoint results
        results['natural'][checkpoint] = {}
        results['icl'][checkpoint] = {}
        results['successful_icl'][checkpoint] = {}
        results['no_cycle_icl'][checkpoint] = {}
        
        # Process each layer
        for layer_dir in sorted(checkpoint_dir.glob("layer_*")):
            layer_idx = int(layer_dir.name.split("_")[1])
            
            # Find the output log file
            log_file = layer_dir / f"full_analysis_cyc{n_cycles}_ml{max_length}.out"
            
            if not log_file.exists():
                print(f"Warning: Log file {log_file} does not exist")
                continue
            
            # Parse the log file (now returns 4 values)
            natural_heatmap, icl_heatmap, successful_icl_heatmap, no_cycle_icl_heatmap = parse_multihead_log(log_file, layer_idx)
            
            if natural_heatmap is not None:
                results['natural'][checkpoint][layer_idx] = natural_heatmap
            if icl_heatmap is not None:
                results['icl'][checkpoint][layer_idx] = icl_heatmap
            if successful_icl_heatmap is not None:
                results['successful_icl'][checkpoint][layer_idx] = successful_icl_heatmap
            if no_cycle_icl_heatmap is not None:
                results['no_cycle_icl'][checkpoint][layer_idx] = no_cycle_icl_heatmap
                
        print(f"Loaded {len(results['natural'][checkpoint])} layers for checkpoint {checkpoint}")
    
    return results


def parse_multihead_log(log_file, expected_layer):
    """
    Parse a single log file to extract multi-head heatmap data
    
    Args:
        log_file: Path to the log file
        expected_layer: Expected layer index for validation
        
    Returns:
        tuple: (natural_heatmap, icl_heatmap, successful_icl_heatmap) as numpy arrays or None
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract natural heatmap
    natural_pattern = rf"layer {expected_layer} natural heatmap: (.+)"
    natural_match = re.search(natural_pattern, content)
    natural_heatmap = None
    
    if natural_match:
        heatmap_str = natural_match.group(1).strip()
        if heatmap_str != "None":
            try:
                # The heatmap should now contain 16 values (one per head)
                natural_heatmap = np.array(eval(heatmap_str))
                print(f"Layer {expected_layer} natural heatmap shape: {natural_heatmap.shape}")
            except Exception as e:
                print(f"Error parsing natural heatmap for layer {expected_layer}: {e}")
    
    # Extract ICL heatmap  
    icl_pattern = rf"layer {expected_layer} icl heatmap: (.+)"
    icl_match = re.search(icl_pattern, content)
    icl_heatmap = None
    
    if icl_match:
        heatmap_str = icl_match.group(1).strip()
        if heatmap_str != "None":
            try:
                icl_heatmap = np.array(eval(heatmap_str))
                print(f"Layer {expected_layer} ICL heatmap shape: {icl_heatmap.shape}")
            except Exception as e:
                print(f"Error parsing ICL heatmap for layer {expected_layer}: {e}")
    
    # Extract successful ICL heatmap (new logging format)
    successful_icl_pattern = rf"layer {expected_layer} successful icl heatmap: (.+)"
    successful_icl_match = re.search(successful_icl_pattern, content)
    successful_icl_heatmap = None
    
    if successful_icl_match:
        heatmap_str = successful_icl_match.group(1).strip()
        if heatmap_str != "None":
            try:
                successful_icl_heatmap = np.array(eval(heatmap_str))
                print(f"Layer {expected_layer} successful ICL heatmap shape: {successful_icl_heatmap.shape}")
            except Exception as e:
                print(f"Error parsing successful ICL heatmap for layer {expected_layer}: {e}")
    
    # Extract no-cycle ICL heatmap (new logging format)
    no_cycle_icl_pattern = rf"layer {expected_layer} no-cycle icl heatmap: (.+)"
    no_cycle_icl_match = re.search(no_cycle_icl_pattern, content)
    no_cycle_icl_heatmap = None
    
    if no_cycle_icl_match:
        heatmap_str = no_cycle_icl_match.group(1).strip()
        if heatmap_str != "None":
            try:
                no_cycle_icl_heatmap = np.array(eval(heatmap_str))
                print(f"Layer {expected_layer} no-cycle ICL heatmap shape: {no_cycle_icl_heatmap.shape}")
            except Exception as e:
                print(f"Error parsing no-cycle ICL heatmap for layer {expected_layer}: {e}")
    
    return natural_heatmap, icl_heatmap, successful_icl_heatmap, no_cycle_icl_heatmap


def parse_repetition_data(log_file, expected_layer):
    """
    Parse repetition data from log file to extract data indices and repetition indices
    
    Args:
        log_file: Path to the log file
        expected_layer: Expected layer index for validation
        
    Returns:
        tuple: (data_indices, repetition_indices) as numpy arrays or None
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract data index
    data_pattern = rf"layer {expected_layer} data index: \[(.+?)\]"
    data_match = re.search(data_pattern, content)
    data_indices = None
    
    if data_match:
        try:
            data_str = data_match.group(1).strip()
            data_indices = np.array([int(x.strip()) for x in data_str.split(',')])
        except Exception as e:
            print(f"Error parsing data indices for layer {expected_layer}: {e}")
    
    # Extract repetition index
    rep_pattern = rf"layer {expected_layer} repetition index: \[(.+?)\]"
    rep_match = re.search(rep_pattern, content)
    repetition_indices = None
    
    if rep_match:
        try:
            rep_str = rep_match.group(1).strip()
            repetition_indices = np.array([int(x.strip()) for x in rep_str.split(',')])
        except Exception as e:
            print(f"Error parsing repetition indices for layer {expected_layer}: {e}")
    
    return data_indices, repetition_indices


def load_repetition_evolution(base_path, model_name="EleutherAI/pythia-1.4b", checkpoints=None, 
                             n_cycles=5, max_length=32, target_layer=20):
    """
    Load repetition evolution data across checkpoints for a specific layer
    
    Args:
        base_path: Base output directory
        model_name: Model name
        checkpoints: List of checkpoints to analyze
        n_cycles: Number of cycles
        max_length: Maximum sequence length
        target_layer: Which layer to analyze repetition patterns for
        
    Returns:
        dict: Repetition data organized by checkpoint
    """
    if checkpoints is None:
        checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    
    base_dir = Path(base_path) / model_name
    repetition_data = {}
    
    for checkpoint in checkpoints:
        checkpoint_dir = base_dir / checkpoint / f"layer_{target_layer}"
        log_file = checkpoint_dir / f"full_analysis_cyc{n_cycles}_ml{max_length}.out"
        
        if not log_file.exists():
            print(f"Warning: Log file {log_file} does not exist")
            continue
        
        data_indices, repetition_indices = parse_repetition_data(log_file, target_layer)
        
        if data_indices is not None and repetition_indices is not None:
            repetition_data[checkpoint] = {
                'data_indices': data_indices,
                'repetition_indices': repetition_indices,
                'total_samples': len(data_indices)
            }
            print(f"Loaded repetition data for {checkpoint}: {len(repetition_indices)} repeating out of {len(data_indices)} total")
    
    return repetition_data


def create_alluvial_plot(repetition_data, save_path=None, target_layer=20, alluvial_only=False, paper_ready=False):
    """
    Create an improved alluvial plot showing how datapoints transition between 
    repeating and non-repeating states across training checkpoints
    
    Args:
        repetition_data: Dictionary from load_repetition_evolution
        save_path: Path to save the plot
        target_layer: Layer number for plot title
        alluvial_only: If True, only create alluvial plot without other analysis
        paper_ready: If True, use publication-quality styling
    """
    
    # Configure paper-ready styling
    if paper_ready:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif', 
            'font.serif': ['Times New Roman'],
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 6,
            'xtick.minor.size': 3,
            'ytick.major.size': 6,
            'ytick.minor.size': 3,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    checkpoints = list(repetition_data.keys())
    if len(checkpoints) < 2:
        print("Need at least 2 checkpoints for alluvial plot")
        return
    
    # Create transition data
    all_datapoints = set()
    for checkpoint_data in repetition_data.values():
        all_datapoints.update(checkpoint_data['data_indices'])
    
    all_datapoints = sorted(list(all_datapoints))
    n_datapoints = len(all_datapoints)
    
    # Track states across checkpoints
    datapoint_states = {}  # checkpoint -> datapoint -> is_repeating
    
    for checkpoint in checkpoints:
        if checkpoint not in repetition_data:
            continue
            
        data_indices = repetition_data[checkpoint]['data_indices']
        rep_indices = repetition_data[checkpoint]['repetition_indices']
        repeating_set = set(rep_indices)
        
        datapoint_states[checkpoint] = {}
        
        for dp in all_datapoints:
            is_repeating = dp in repeating_set
            datapoint_states[checkpoint][dp] = is_repeating
    
    # Progressive categorization: only split when transitions occur
    def get_categories_for_checkpoint(checkpoint_idx):
        """Get categories available up to this checkpoint"""
        if checkpoint_idx == 0:
            return ["repeating", "non_repeating"]
        
        categories = ["non_repeating"]  # Always have non-repeating
        
        # Add categories for each checkpoint where datapoints first became repeating
        for i in range(checkpoint_idx + 1):
            cp = checkpoints[i]
            categories.append(f"repeating_since_{cp}")
        
        return categories
    
    # Assign progressive categories to datapoints
    datapoint_progressive_categories = {}  # checkpoint -> datapoint -> category
    
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint not in datapoint_states:
            continue
        
        datapoint_progressive_categories[checkpoint] = {}
        
        for dp in all_datapoints:
            if dp not in datapoint_states[checkpoint]:
                continue
                
            is_repeating = datapoint_states[checkpoint][dp]
            
            if not is_repeating:
                datapoint_progressive_categories[checkpoint][dp] = "non_repeating"
            else:
                # For first checkpoint, all repeating datapoints use simple "repeating" category
                if i == 0:
                    datapoint_progressive_categories[checkpoint][dp] = "repeating"
                else:
                    # Find when this datapoint first became repeating
                    first_repeating_checkpoint = None
                    for j in range(i + 1):  # Check up to current checkpoint
                        prev_cp = checkpoints[j]
                        if (prev_cp in datapoint_states and 
                            dp in datapoint_states[prev_cp] and 
                            datapoint_states[prev_cp][dp]):
                            first_repeating_checkpoint = prev_cp
                            break
                    
                    if first_repeating_checkpoint:
                        datapoint_progressive_categories[checkpoint][dp] = f"repeating_since_{first_repeating_checkpoint}"
                    else:
                        datapoint_progressive_categories[checkpoint][dp] = f"repeating_since_{checkpoint}"
    
    # Count datapoints by progressive category at each checkpoint
    progressive_category_counts = {}
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint not in datapoint_progressive_categories:
            continue
        
        available_categories = get_categories_for_checkpoint(i)
        progressive_category_counts[checkpoint] = {}
        
        # Initialize all available categories
        for category in available_categories:
            progressive_category_counts[checkpoint][category] = {
                'repeating': 0,
                'non_repeating': 0
            }
        
        # Count actual datapoints
        for dp in all_datapoints:
            if (dp in datapoint_progressive_categories[checkpoint] and 
                dp in datapoint_states[checkpoint]):
                
                category = datapoint_progressive_categories[checkpoint][dp]
                is_repeating = datapoint_states[checkpoint][dp]
                
                # Make sure the category exists in our counts
                if category not in progressive_category_counts[checkpoint]:
                    progressive_category_counts[checkpoint][category] = {
                        'repeating': 0,
                        'non_repeating': 0
                    }
                
                if is_repeating:
                    progressive_category_counts[checkpoint][category]['repeating'] += 1
                else:
                    progressive_category_counts[checkpoint][category]['non_repeating'] += 1
    
    # Create color mapping - paper-appropriate colors
    def get_color_for_category(category, checkpoint_idx):
        if paper_ready:
            # Pastel color palette for papers - beautiful and paper-ready
            paper_colors = {
                "non_repeating": '#F0F0F0',      # Much lighter grey
                "repeating": '#8FB3D3',          # Soft pastel blue
                "repeating_since_step1": '#8FB3D3',       # Soft blue
                "repeating_since_step1000": '#C8A2C8',    # Soft lavender
                "repeating_since_step5000": '#FFB366',    # Soft peach
                "repeating_since_step10000": '#F2A0A0',   # Soft coral
                "repeating_since_step100000": '#A8D8A8',  # Soft sage green
                "repeating_since_steplatest": '#D2B3A0'   # Soft taupe
            }
            
            if category in paper_colors:
                return paper_colors[category]
            elif category.startswith("repeating_since_"):
                # Fallback gradient for additional checkpoints - pastel tones
                cp_name = category.replace("repeating_since_", "")
                try:
                    cp_idx = checkpoints.index(cp_name)
                    gradient_colors = ['#8FB3D3', '#C8A2C8', '#FFB366', '#F2A0A0', '#A8D8A8', '#D2B3A0']
                    return gradient_colors[cp_idx % len(gradient_colors)]
                except ValueError:
                    return '#D0D0D0'  # Light gray fallback
            else:
                return '#D0D0D0'  # Light gray fallback
        else:
            # Original color scheme
            if category == "non_repeating":
                return 'lightblue'
            elif category == "repeating":
                return 'red'
            elif category.startswith("repeating_since_"):
                # Use different colors for different "repeating_since_X" categories
                checkpoint_colors = plt.cm.Set3(np.linspace(0, 1, len(checkpoints)))
                # Extract checkpoint name from category
                cp_name = category.replace("repeating_since_", "")
                try:
                    cp_idx = checkpoints.index(cp_name)
                    return checkpoint_colors[cp_idx]
                except ValueError:
                    # Fallback color if checkpoint not found
                    return 'red'
            else:
                # Fallback for any unexpected categories
                return 'gray'
    
    # Create the plot with progressive categorization
    if paper_ready:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig, ax = plt.subplots(figsize=(18, 12))
    
    checkpoint_positions = list(range(len(checkpoints)))
    bar_width = 0.6
    spacing = 1.0
    
    # Calculate total counts for positioning
    total_counts_per_checkpoint = {}
    frontier_positions = {}  # Store y-position of red frontier line
    
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint not in progressive_category_counts:
            total_counts_per_checkpoint[checkpoint] = 0
            frontier_positions[checkpoint] = 0
            continue
        
        total = 0
        non_rep_total = 0
        
        available_categories = get_categories_for_checkpoint(i)
        for category in available_categories:
            rep_count = progressive_category_counts[checkpoint][category]['repeating']
            non_rep_count = progressive_category_counts[checkpoint][category]['non_repeating']
            total += rep_count + non_rep_count
            non_rep_total += non_rep_count
        
        total_counts_per_checkpoint[checkpoint] = total
        frontier_positions[checkpoint] = non_rep_total  # Y-position where repeating starts
    
    max_total = max(total_counts_per_checkpoint.values()) if total_counts_per_checkpoint else 1
    
    # Draw stacked bars with progressive categorization and red frontier
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint not in progressive_category_counts:
            continue
        
        x_pos = i * spacing
        current_y = 0
        available_categories = get_categories_for_checkpoint(i)
        
        # FIRST PASS: Draw all non-repeating parts (bottom layer)
        for category in available_categories:
            non_rep_count = progressive_category_counts[checkpoint][category]['non_repeating']
            
            if non_rep_count > 0:
                color = get_color_for_category(category, i)
                bar_non_rep = ax.bar(x_pos, non_rep_count, bottom=current_y, width=bar_width,
                                   color=color, alpha=0.6,  # Slightly more visible but still light
                                   edgecolor='black', linewidth=0.5)
                
                # Add proportion as text label
                total_at_checkpoint = total_counts_per_checkpoint[checkpoint]
                proportion = non_rep_count / total_at_checkpoint if total_at_checkpoint > 0 else 0
                ax.text(x_pos, current_y + non_rep_count/2, f'{proportion:.2f}',
                       ha='center', va='center', fontweight='bold', fontsize=8)
                
                current_y += non_rep_count
        
        # Draw red frontier line - paper-ready styling
        frontier_y = frontier_positions[checkpoint]
        if paper_ready:
            ax.hlines(y=frontier_y, xmin=x_pos - bar_width/2, xmax=x_pos + bar_width/2, 
                     colors='#B91C1C', linewidth=3, alpha=0.9)  # Professional red
        else:
            ax.hlines(y=frontier_y, xmin=x_pos - bar_width/2, xmax=x_pos + bar_width/2, 
                     colors='red', linewidth=4, alpha=0.9)
        
        # SECOND PASS: Draw all repeating parts (top layer)
        for category in available_categories:
            rep_count = progressive_category_counts[checkpoint][category]['repeating']
            
            if rep_count > 0:
                color = get_color_for_category(category, i)
                bar_rep = ax.bar(x_pos, rep_count, bottom=current_y, width=bar_width,
                               color=color, alpha=0.8, 
                               edgecolor='black', linewidth=0.5)
                
                # Add proportion as text label
                total_at_checkpoint = total_counts_per_checkpoint[checkpoint]
                proportion = rep_count / total_at_checkpoint if total_at_checkpoint > 0 else 0
                ax.text(x_pos, current_y + rep_count/2, f'{proportion:.2f}',
                       ha='center', va='center', fontweight='bold', fontsize=8)
                
                current_y += rep_count
        
    # Add explanatory text AFTER the last column (on the right)
    if len(checkpoints) > 0:
        last_checkpoint = checkpoints[-1]
        if last_checkpoint in total_counts_per_checkpoint and last_checkpoint in frontier_positions:
            total_height = total_counts_per_checkpoint[last_checkpoint]
            frontier_y = frontier_positions[last_checkpoint]
            
            # Position text to the right of the last column
            text_x = (len(checkpoints) - 1) * spacing + bar_width * 1.2
            
            # "NON-REPEATING" text in bottom section
            if frontier_y > 0:
                if paper_ready:
                    ax.text(text_x, frontier_y/2, 'Non-Repeating',
                           rotation=90, ha='center', va='center', 
                           fontweight='bold', fontsize=11, color='#696969')  # Darker gray for visibility
                else:
                    ax.text(text_x, frontier_y/2, 'NON-REPEATING',
                           rotation=90, ha='center', va='center', 
                           fontweight='bold', fontsize=12, color='blue')
            
            # "REPEATING" text in top section
            if total_height > frontier_y:
                if paper_ready:
                    ax.text(text_x, frontier_y + (total_height - frontier_y)/2, 'Repeating',
                           rotation=90, ha='center', va='center', 
                           fontweight='bold', fontsize=11, color='#4682B4')  # Steel blue for better contrast
                else:
                    ax.text(text_x, frontier_y + (total_height - frontier_y)/2, 'REPEATING',
                           rotation=90, ha='center', va='center', 
                           fontweight='bold', fontsize=12, color='red')
    
    # Draw flows between checkpoints with progressive categorization
    for i in range(len(checkpoints) - 1):
        cp1, cp2 = checkpoints[i], checkpoints[i + 1]
        
        if cp1 not in progressive_category_counts or cp2 not in progressive_category_counts:
            continue
        
        x1 = i * spacing + bar_width/2
        x2 = (i + 1) * spacing - bar_width/2
        
        # Get available categories for both checkpoints
        categories1 = get_categories_for_checkpoint(i)
        categories2 = get_categories_for_checkpoint(i + 1)
        
        # Calculate y-positions for cp1 (non-repeating bottom, repeating top)
        y1_non_rep_start = 0
        y1_positions = {}
        
        # Non-repeating positions for cp1
        for category in categories1:
            non_rep1 = progressive_category_counts[cp1][category]['non_repeating']
            y1_positions[f"{category}_non_rep"] = {'start': y1_non_rep_start, 'count': non_rep1}
            y1_non_rep_start += non_rep1
        
        # Repeating positions for cp1
        y1_rep_start = y1_non_rep_start
        for category in categories1:
            rep1 = progressive_category_counts[cp1][category]['repeating']
            y1_positions[f"{category}_rep"] = {'start': y1_rep_start, 'count': rep1}
            y1_rep_start += rep1
        
        # Calculate y-positions for cp2
        y2_non_rep_start = 0
        y2_positions = {}
        
        # Non-repeating positions for cp2
        for category in categories2:
            non_rep2 = progressive_category_counts[cp2][category]['non_repeating']
            y2_positions[f"{category}_non_rep"] = {'start': y2_non_rep_start, 'count': non_rep2}
            y2_non_rep_start += non_rep2
        
        # Repeating positions for cp2
        y2_rep_start = y2_non_rep_start
        for category in categories2:
            rep2 = progressive_category_counts[cp2][category]['repeating']
            y2_positions[f"{category}_rep"] = {'start': y2_rep_start, 'count': rep2}
            y2_rep_start += rep2
        
        # Draw flows based on datapoint transitions with optimized ordering
        # Track individual datapoint transitions to create proper flows
        transitions = {}  # (from_category, to_category) -> count
        
        for dp in all_datapoints:
            if (cp1 not in datapoint_progressive_categories or 
                cp2 not in datapoint_progressive_categories or
                dp not in datapoint_progressive_categories[cp1] or
                dp not in datapoint_progressive_categories[cp2]):
                continue
            
            cat1 = datapoint_progressive_categories[cp1][dp]
            cat2 = datapoint_progressive_categories[cp2][dp]
            state1 = datapoint_states[cp1][dp]
            state2 = datapoint_states[cp2][dp]
            
            from_key = f"{cat1}_{'rep' if state1 else 'non_rep'}"
            to_key = f"{cat2}_{'rep' if state2 else 'non_rep'}"
            
            transition_key = (from_key, to_key)
            transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        # Sort transitions to prioritize flows going to repeating sections
        # This ensures flows from non-repeating to repeating leave from the top
        def transition_priority(transition_item):
            from_key, to_key = transition_item[0]
            # Priority 1: non-rep to rep (should go from top of non-rep section)
            if 'non_rep' in from_key and 'rep' in to_key:
                return 0
            # Priority 2: rep to rep (should go from top)
            elif 'rep' in from_key and 'rep' in to_key:
                return 1
            # Priority 3: rep to non-rep (goes to bottom)
            elif 'rep' in from_key and 'non_rep' in to_key:
                return 2
            # Priority 4: non-rep to non-rep (stays in bottom)
            else:
                return 3
        
        sorted_transitions = sorted(transitions.items(), key=transition_priority)
        
        # For non-repeating sources, we need to track from the top down
        # For repeating sources, we track from bottom up as usual
        non_rep_positions_top = {}  # Track from top for non-rep sources going to rep
        rep_positions_top = {}      # Track from top for rep sources going upward
        
        # Initialize top-down tracking for non-repeating sections
        for from_key in y1_positions:
            if 'non_rep' in from_key:
                # Start from the top of the non-repeating section
                top_position = y1_positions[from_key]['start'] + y1_positions[from_key]['count']
                non_rep_positions_top[from_key] = top_position
            elif 'rep' in from_key:
                # Start from the top of the repeating section for upward flows
                top_position = y1_positions[from_key]['start'] + y1_positions[from_key]['count']
                rep_positions_top[from_key] = top_position
        
        # Helper function to determine if a flow is going upward
        def is_upward_flow(from_key, to_key):
            # Get the y-positions to determine direction
            from_y_center = y1_positions[from_key]['start'] + y1_positions[from_key]['count'] / 2
            to_y_center = y2_positions[to_key]['start'] + y2_positions[to_key]['count'] / 2
            return to_y_center > from_y_center
        
        # Draw flow polygons for each transition in optimized order
        for (from_key, to_key), count in sorted_transitions:
            if count == 0:
                continue
            
            if from_key not in y1_positions or to_key not in y2_positions:
                continue
            
            # Get flow color - use the color of the DESTINATION category/state
            to_category_name = to_key.replace('_rep', '').replace('_non_rep', '')
            to_state = 'rep' if '_rep' in to_key else 'non_rep'
            
            # Get the actual destination color from the same function used for bars
            flow_color = get_color_for_category(to_category_name, i + 1)
            
            # Adjust alpha based on destination state (match the bar alpha)
            if to_state == 'rep':
                alpha = 0.8  # Match repeating bar alpha
            else:
                alpha = 0.6  # Match non-repeating bar alpha (updated)
            
            # Special styling for state changes with softer colors for non-repeating flows
            edge_style = '-'
            edge_color = 'black'
            edge_width = 0.3
            
            if 'non_rep' in from_key and 'rep' in to_key:
                edge_style = '--'
                edge_width = 0.8
            elif 'rep' in from_key and 'non_rep' in to_key:
                edge_style = ':'
                edge_width = 0.8
            
            # Calculate flow polygon vertices with optimized positioning
            # Check if this is an upward flow and start from the top accordingly
            if is_upward_flow(from_key, to_key):
                # ALL upward flows start from the top of their source column
                if 'non_rep' in from_key:
                    # Non-repeating source going upward
                    y1_end = non_rep_positions_top[from_key]
                    y1_start = y1_end - count
                    non_rep_positions_top[from_key] = y1_start  # Update for next flow
                else:
                    # Repeating source going upward
                    y1_end = rep_positions_top[from_key]
                    y1_start = y1_end - count
                    rep_positions_top[from_key] = y1_start  # Update for next flow
            else:
                # Downward or horizontal flows: start from bottom as usual
                y1_start = y1_positions[from_key]['start']
                y1_end = y1_start + count
                y1_positions[from_key]['start'] += count
            
            # Destination positioning remains standard
            y2_start = y2_positions[to_key]['start']
            y2_end = y2_start + count
            y2_positions[to_key]['start'] += count
            
            flow_vertices = [
                [x1, y1_start], [x1, y1_end],
                [x2, y2_end], [x2, y2_start]
            ]
            
            flow_poly = Polygon(flow_vertices, alpha=alpha,
                              color=flow_color,
                              edgecolor=edge_color,
                              linewidth=edge_width,
                              linestyle=edge_style)
            ax.add_patch(flow_poly)
    
    # Formatting
    ax.set_xlim(-bar_width * 0.5, len(checkpoints) * spacing + bar_width * 1.5)  # Extra space for right text
    ax.set_ylim(0, max_total * 1.1)
    ax.set_xticks([i * spacing for i in range(len(checkpoints))])
    ax.set_xticklabels(checkpoints, rotation=45, ha='right')
    
    if paper_ready:
        ax.set_ylabel('Number of Datapoints', fontsize=13, fontweight='bold')
        ax.set_title(f'Repetition Evolution Across Training\n(Layer {target_layer})', 
                    fontsize=14, fontweight='bold', pad=20)
        # Clean grid for papers
        ax.grid(True, alpha=0.2, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor('#fafafa')
    else:
        ax.set_ylabel('Number of Datapoints', fontsize=14)
        ax.set_title(f'Alluvial Plot: Progressive Repetition Evolution\n(Layer {target_layer})', 
                    fontsize=16, pad=20)
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        if paper_ready:
            # Save both PNG and PDF for papers
            base_path = save_path.replace('.png', '').replace('.pdf', '')
            plt.savefig(f"{base_path}_paper.png", bbox_inches='tight', dpi=300)
            plt.savefig(f"{base_path}_paper.pdf", bbox_inches='tight', dpi=300)
            print(f"Saved paper-ready alluvial plot to {base_path}_paper.png and .pdf")
        else:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved improved alluvial plot to {save_path}")
    
    plt.show()
    
    # Print detailed summary statistics
    if not alluvial_only:
        print(f"\nProgressive Repetition Evolution Summary (Layer {target_layer}):")
        print("=" * 80)
        
        for i, checkpoint in enumerate(checkpoints):
            if checkpoint not in progressive_category_counts:
                continue
            
            available_categories = get_categories_for_checkpoint(i)
            print(f"\n{checkpoint} (Categories: {len(available_categories)}):")
            
            total_rep = sum(progressive_category_counts[checkpoint][cat]['repeating'] for cat in available_categories)
            total_non_rep = sum(progressive_category_counts[checkpoint][cat]['non_repeating'] for cat in available_categories)
            total = total_rep + total_non_rep
            
            print(f"  Total: {total} datapoints")
            print(f"  Repeating: {total_rep} ({total_rep/total*100:.1f}%)")
            print(f"  Non-repeating: {total_non_rep} ({total_non_rep/total*100:.1f}%)")
            
            print("  By category:")
            for category in available_categories:
                rep = progressive_category_counts[checkpoint][category]['repeating']
                non_rep = progressive_category_counts[checkpoint][category]['non_repeating']
                if rep + non_rep > 0:
                    cat_name = category.replace("repeating_since_", "Rep@") if "repeating_since_" in category else category
                    print(f"    {cat_name}: {rep}R + {non_rep}NR = {rep + non_rep}")


def create_plotly_alluvial_plot(repetition_data, save_path=None, target_layer=20):
    """
    Create a Plotly-based Sankey diagram for repetition evolution
    Requires: pip install plotly
    
    Args:
        repetition_data: Dictionary from load_repetition_evolution
        save_path: Path to save the plot (will save as HTML)
        target_layer: Layer number for plot title
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        print("Falling back to matplotlib version...")
        return create_alluvial_plot(repetition_data, save_path, target_layer)
    
    checkpoints = list(repetition_data.keys())
    if len(checkpoints) < 2:
        print("Need at least 2 checkpoints for alluvial plot")
        return
    
    # Create transition data similar to matplotlib version
    all_datapoints = set()
    for checkpoint_data in repetition_data.values():
        all_datapoints.update(checkpoint_data['data_indices'])
    
    all_datapoints = sorted(list(all_datapoints))
    
    # Track states and categories
    first_repeating_checkpoint = {}
    datapoint_states = {}
    datapoint_categories = {}
    
    for checkpoint in checkpoints:
        if checkpoint not in repetition_data:
            continue
            
        data_indices = repetition_data[checkpoint]['data_indices']
        rep_indices = repetition_data[checkpoint]['repetition_indices']
        repeating_set = set(rep_indices)
        
        datapoint_states[checkpoint] = {}
        
        for dp in all_datapoints:
            is_repeating = dp in repeating_set
            datapoint_states[checkpoint][dp] = is_repeating
            
            if is_repeating and dp not in first_repeating_checkpoint:
                first_repeating_checkpoint[dp] = checkpoint
                datapoint_categories[dp] = f"first_rep_{checkpoint}"
    
    for dp in all_datapoints:
        if dp not in datapoint_categories:
            datapoint_categories[dp] = "never_repeating"
    
    # Create nodes and links for Sankey diagram
    nodes = []
    node_colors = []
    links = {'source': [], 'target': [], 'value': [], 'color': []}
    
    # Node naming: checkpoint_category_state
    unique_categories = sorted(list(set(datapoint_categories.values())))
    category_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
    
    node_index = 0
    node_mapping = {}
    
    # Create nodes for each checkpoint-category-state combination
    for i, checkpoint in enumerate(checkpoints):
        for category in unique_categories:
            for state in ['repeating', 'non_repeating']:
                node_name = f"{checkpoint}_{category}_{state}"
                nodes.append(f"{checkpoint}<br>{category.replace('first_rep_', 'First@')}<br>({state})")
                
                # Color based on category
                cat_idx = unique_categories.index(category)
                alpha = 0.8 if state == 'repeating' else 0.4
                color = f"rgba({int(category_colors[cat_idx][0]*255)},{int(category_colors[cat_idx][1]*255)},{int(category_colors[cat_idx][2]*255)},{alpha})"
                node_colors.append(color)
                
                node_mapping[node_name] = node_index
                node_index += 1
    
    # Create links between consecutive checkpoints
    for i in range(len(checkpoints) - 1):
        cp1, cp2 = checkpoints[i], checkpoints[i + 1]
        
        if cp1 not in datapoint_states or cp2 not in datapoint_states:
            continue
        
        # Count transitions by category
        for category in unique_categories:
            datapoints_in_category = [dp for dp, cat in datapoint_categories.items() if cat == category]
            
            if not datapoints_in_category:
                continue
            
            # Count transitions
            rep_to_rep = 0
            rep_to_non = 0
            non_to_rep = 0
            non_to_non = 0
            
            for dp in datapoints_in_category:
                if dp not in datapoint_states[cp1] or dp not in datapoint_states[cp2]:
                    continue
                
                state1 = datapoint_states[cp1][dp]
                state2 = datapoint_states[cp2][dp]
                
                if state1 and state2:
                    rep_to_rep += 1
                elif state1 and not state2:
                    rep_to_non += 1
                elif not state1 and state2:
                    non_to_rep += 1
                else:
                    non_to_non += 1
            
            # Add links
            cat_idx = unique_categories.index(category)
            link_color = f"rgba({int(category_colors[cat_idx][0]*255)},{int(category_colors[cat_idx][1]*255)},{int(category_colors[cat_idx][2]*255)},0.3)"
            
            transitions = [
                (f"{cp1}_{category}_repeating", f"{cp2}_{category}_repeating", rep_to_rep),
                (f"{cp1}_{category}_repeating", f"{cp2}_{category}_non_repeating", rep_to_non),
                (f"{cp1}_{category}_non_repeating", f"{cp2}_{category}_repeating", non_to_rep),
                (f"{cp1}_{category}_non_repeating", f"{cp2}_{category}_non_repeating", non_to_non)
            ]
            
            for source_node, target_node, value in transitions:
                if value > 0:
                    links['source'].append(node_mapping[source_node])
                    links['target'].append(node_mapping[target_node])
                    links['value'].append(value)
                    links['color'].append(link_color)
    
    # Create Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    ))
    
    fig.update_layout(
        title_text=f"Repetition Evolution Sankey Diagram<br>Layer {target_layer}",
        font_size=12,
        width=1200,
        height=800
    )
    
    if save_path:
        html_path = save_path.replace('.png', '.html') if save_path.endswith('.png') else f"{save_path}.html"
        fig.write_html(html_path)
        print(f"Saved Plotly Sankey diagram to {html_path}")
    
    fig.show()
    
    return fig


def create_multihead_dataframe(results, heatmap_type='natural'):
    """
    Convert multi-head results into a pandas DataFrame for analysis
    
    Args:
        results: Results dictionary from load_multihead_results
        heatmap_type: 'natural', 'icl', or 'successful_icl'
        
    Returns:
        pd.DataFrame: With columns ['checkpoint', 'layer', 'head', 'contrast']
    """
    data_rows = []
    
    for checkpoint, layers in results[heatmap_type].items():
        for layer_idx, head_values in layers.items():
            if head_values is not None:
                for head_idx, contrast_value in enumerate(head_values):
                    data_rows.append({
                        'checkpoint': checkpoint,
                        'layer': layer_idx,
                        'head': head_idx,
                        'contrast': float(contrast_value),
                        'layer_head': f"{layer_idx}.{head_idx}",
                        'category': heatmap_type
                    })
    
    return pd.DataFrame(data_rows)


def plot_multihead_evolution(results, save_path=None):
    """
    Create plots showing how attention head contrasts evolve across checkpoints
    
    Args:
        results: Results dictionary from load_multihead_results
        save_path: Path to save the plot (optional)
    """
    # Create DataFrames
    natural_df = create_multihead_dataframe(results, 'natural')
    icl_df = create_multihead_dataframe(results, 'icl')
    successful_icl_df = create_multihead_dataframe(results, 'successful_icl') if 'successful_icl' in results else pd.DataFrame()
    no_cycle_icl_df = create_multihead_dataframe(results, 'no_cycle_icl') if 'no_cycle_icl' in results else pd.DataFrame()
    
    if natural_df.empty and icl_df.empty and successful_icl_df.empty and no_cycle_icl_df.empty:
        print("No data available for plotting")
        return
    
    # Find top performing heads across all checkpoints
    if not natural_df.empty:
        top_natural_heads = natural_df.groupby('layer_head')['contrast'].mean().nlargest(3).index
        bottom_natural_heads = natural_df.groupby('layer_head')['contrast'].mean().nsmallest(3).index
        interesting_natural_heads = list(top_natural_heads) + list(bottom_natural_heads)
    else:
        interesting_natural_heads = []
    
    if not icl_df.empty:
        top_icl_heads = icl_df.groupby('layer_head')['contrast'].mean().nlargest(3).index  
        bottom_icl_heads = icl_df.groupby('layer_head')['contrast'].mean().nsmallest(3).index
        interesting_icl_heads = list(top_icl_heads) + list(bottom_icl_heads)
    else:
        interesting_icl_heads = []
    
    if not successful_icl_df.empty:
        top_successful_icl_heads = successful_icl_df.groupby('layer_head')['contrast'].mean().nlargest(3).index  
        bottom_successful_icl_heads = successful_icl_df.groupby('layer_head')['contrast'].mean().nsmallest(3).index
        interesting_successful_icl_heads = list(top_successful_icl_heads) + list(bottom_successful_icl_heads)
    else:
        interesting_successful_icl_heads = []
    
    if not no_cycle_icl_df.empty:
        top_no_cycle_icl_heads = no_cycle_icl_df.groupby('layer_head')['contrast'].mean().nlargest(3).index  
        bottom_no_cycle_icl_heads = no_cycle_icl_df.groupby('layer_head')['contrast'].mean().nsmallest(3).index
        interesting_no_cycle_icl_heads = list(top_no_cycle_icl_heads) + list(bottom_no_cycle_icl_heads)
    else:
        interesting_no_cycle_icl_heads = []
    
    # Calculate shared y-axis limits for consistent scaling
    all_contrast_values = []
    if not natural_df.empty:
        all_contrast_values.extend(natural_df['contrast'].values)
    if not icl_df.empty:
        all_contrast_values.extend(icl_df['contrast'].values)
    if not successful_icl_df.empty:
        all_contrast_values.extend(successful_icl_df['contrast'].values)
    if not no_cycle_icl_df.empty:
        all_contrast_values.extend(no_cycle_icl_df['contrast'].values)
    
    if all_contrast_values:
        y_min = min(all_contrast_values)
        y_max = max(all_contrast_values)
        # Add some padding (5% on each side)
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        y_min_padded = y_min - y_padding
        y_max_padded = y_max + y_padding
    else:
        y_min_padded, y_max_padded = 0, 1  # fallback values
    
    # Create subplots (up to 4 columns)
    n_cols = 2 + int(not successful_icl_df.empty) + int(not no_cycle_icl_df.empty)
    fig, axes = plt.subplots(1, n_cols, figsize=(8*n_cols, 8))
    if n_cols == 2:
        axes = [axes[0], axes[1]]
    
    col = 0
    # Plot natural contrasts
    if not natural_df.empty:
        sns.lineplot(data=natural_df, x="checkpoint", y="contrast", hue="layer_head", 
                    alpha=0.1, legend=False, ax=axes[col], color="grey")
        interesting_natural_df = natural_df[natural_df['layer_head'].isin(interesting_natural_heads)]
        sns.lineplot(data=interesting_natural_df, x="checkpoint", y="contrast", 
                    hue="layer_head", style="layer_head", markers=True, 
                    markersize=10, linewidth=3, ax=axes[col])
        axes[col].set_title(f"Natural Contrast Evolution\n{results['metadata']['model_name']}", fontsize=14)
        axes[col].set_xlabel("Training Checkpoint", fontsize=12)
        axes[col].set_ylabel("Contrast", fontsize=12)
        axes[col].tick_params(axis='x', rotation=45)
        axes[col].set_ylim(y_min_padded, y_max_padded)
        axes[col].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        col += 1
    
    # Plot ICL contrasts
    if not icl_df.empty:
        sns.lineplot(data=icl_df, x="checkpoint", y="contrast", hue="layer_head",
                    alpha=0.1, legend=False, ax=axes[col], color="grey")
        interesting_icl_df = icl_df[icl_df['layer_head'].isin(interesting_icl_heads)]
        sns.lineplot(data=interesting_icl_df, x="checkpoint", y="contrast",
                    hue="layer_head", style="layer_head", markers=True,
                    markersize=10, linewidth=3, ax=axes[col])
        axes[col].set_title(f"ICL Contrast Evolution\n{results['metadata']['model_name']}", fontsize=14)
        axes[col].set_xlabel("Training Checkpoint", fontsize=12)
        axes[col].set_ylabel("Contrast", fontsize=12)
        axes[col].tick_params(axis='x', rotation=45)
        axes[col].set_ylim(y_min_padded, y_max_padded)
        axes[col].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        col += 1
    
    # Plot Successful ICL contrasts (if available)
    if not successful_icl_df.empty:
        sns.lineplot(data=successful_icl_df, x="checkpoint", y="contrast", hue="layer_head",
                    alpha=0.1, legend=False, ax=axes[col], color="grey")
        interesting_successful_icl_df = successful_icl_df[successful_icl_df['layer_head'].isin(interesting_successful_icl_heads)]
        sns.lineplot(data=interesting_successful_icl_df, x="checkpoint", y="contrast",
                    hue="layer_head", style="layer_head", markers=True,
                    markersize=10, linewidth=3, ax=axes[col])
        axes[col].set_title(f"Successful ICL Contrast Evolution\n{results['metadata']['model_name']}", fontsize=14)
        axes[col].set_xlabel("Training Checkpoint", fontsize=12)
        axes[col].set_ylabel("Contrast", fontsize=12)
        axes[col].tick_params(axis='x', rotation=45)
        axes[col].set_ylim(y_min_padded, y_max_padded)
        axes[col].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        col += 1
    
    # Plot No-cycle ICL contrasts (if available)
    if not no_cycle_icl_df.empty:
        sns.lineplot(data=no_cycle_icl_df, x="checkpoint", y="contrast", hue="layer_head",
                    alpha=0.1, legend=False, ax=axes[col], color="grey")
        interesting_no_cycle_icl_df = no_cycle_icl_df[no_cycle_icl_df['layer_head'].isin(interesting_no_cycle_icl_heads)]
        sns.lineplot(data=interesting_no_cycle_icl_df, x="checkpoint", y="contrast",
                    hue="layer_head", style="layer_head", markers=True,
                    markersize=10, linewidth=3, ax=axes[col])
        axes[col].set_title(f"No-cycle ICL Contrast Evolution\n{results['metadata']['model_name']}", fontsize=14)
        axes[col].set_xlabel("Training Checkpoint", fontsize=12)
        axes[col].set_ylabel("Contrast", fontsize=12)
        axes[col].tick_params(axis='x', rotation=45)
        axes[col].set_ylim(y_min_padded, y_max_padded)
        axes[col].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        col += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_checkpoint_heatmaps(results, checkpoint="steplatest", save_path=None):
    """
    Create heatmaps showing all layer.head contrasts for a specific checkpoint
    
    Args:
        results: Results dictionary from load_multihead_results
        checkpoint: Which checkpoint to visualize
        save_path: Path to save the plot (optional)
    """
    if checkpoint not in results['natural']:
        print(f"Checkpoint {checkpoint} not found in results")
        return
    
    # Check what data types are available
    has_successful_icl = 'successful_icl' in results and checkpoint in results['successful_icl']
    has_no_cycle_icl = 'no_cycle_icl' in results and checkpoint in results['no_cycle_icl']
    n_cols = 2 + int(has_successful_icl) + int(has_no_cycle_icl)
    
    # Prepare data matrices
    max_layers = max(max(results['natural'][checkpoint].keys(), default=0),
                    max(results['icl'][checkpoint].keys(), default=0)) + 1
    if has_successful_icl:
        max_layers = max(max_layers, max(results['successful_icl'][checkpoint].keys(), default=0) + 1)
    if has_no_cycle_icl:
        max_layers = max(max_layers, max(results['no_cycle_icl'][checkpoint].keys(), default=0) + 1)
    
    num_heads = 16
    
    natural_matrix = np.full((max_layers, num_heads), np.nan)
    icl_matrix = np.full((max_layers, num_heads), np.nan)
    successful_icl_matrix = np.full((max_layers, num_heads), np.nan) if has_successful_icl else None
    no_cycle_icl_matrix = np.full((max_layers, num_heads), np.nan) if has_no_cycle_icl else None
    
    # Fill natural matrix
    for layer_idx, head_values in results['natural'][checkpoint].items():
        if head_values is not None and len(head_values) >= num_heads:
            natural_matrix[layer_idx, :] = head_values[:num_heads]
    
    # Fill ICL matrix
    for layer_idx, head_values in results['icl'][checkpoint].items():
        if head_values is not None and len(head_values) >= num_heads:
            icl_matrix[layer_idx, :] = head_values[:num_heads]
    
    # Fill successful ICL matrix if available
    if has_successful_icl:
        for layer_idx, head_values in results['successful_icl'][checkpoint].items():
            if head_values is not None and len(head_values) >= num_heads:
                successful_icl_matrix[layer_idx, :] = head_values[:num_heads]
    
    # Fill no-cycle ICL matrix if available
    if has_no_cycle_icl:
        for layer_idx, head_values in results['no_cycle_icl'][checkpoint].items():
            if head_values is not None and len(head_values) >= num_heads:
                no_cycle_icl_matrix[layer_idx, :] = head_values[:num_heads]
    
    # Create heatmap plots
    fig, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols, 10))
    if n_cols == 2:
        axes = [axes[0], axes[1]]  # Ensure axes is a list
    elif n_cols == 1:
        axes = [axes]
    
    col = 0
    # Natural heatmap
    sns.heatmap(natural_matrix, cmap="RdBu_r", center=0, 
               xticklabels=range(num_heads), yticklabels=range(max_layers),
               cbar_kws={'label': 'Contrast'}, ax=axes[col])
    axes[col].set_title(f"Natural Contrasts - {checkpoint}\n{results['metadata']['model_name']}", fontsize=14)
    axes[col].set_xlabel("Attention Head", fontsize=12)
    axes[col].set_ylabel("Layer", fontsize=12)
    col += 1
    
    # ICL heatmap
    sns.heatmap(icl_matrix, cmap="RdBu_r", center=0,
               xticklabels=range(num_heads), yticklabels=range(max_layers), 
               cbar_kws={'label': 'Contrast'}, ax=axes[col])
    axes[col].set_title(f"ICL Contrasts - {checkpoint}\n{results['metadata']['model_name']}", fontsize=14)
    axes[col].set_xlabel("Attention Head", fontsize=12)
    axes[col].set_ylabel("Layer", fontsize=12)
    col += 1
    
    # Successful ICL heatmap (if available)
    if has_successful_icl:
        sns.heatmap(successful_icl_matrix, cmap="RdBu_r", center=0,
                   xticklabels=range(num_heads), yticklabels=range(max_layers), 
                   cbar_kws={'label': 'Contrast'}, ax=axes[col])
        axes[col].set_title(f"Successful ICL Contrasts - {checkpoint}\n{results['metadata']['model_name']}", fontsize=14)
        axes[col].set_xlabel("Attention Head", fontsize=12)
        axes[col].set_ylabel("Layer", fontsize=12)
        col += 1
    
    # No-cycle ICL heatmap (if available)
    if has_no_cycle_icl:
        sns.heatmap(no_cycle_icl_matrix, cmap="RdBu_r", center=0,
                   xticklabels=range(num_heads), yticklabels=range(max_layers), 
                   cbar_kws={'label': 'Contrast'}, ax=axes[col])
        axes[col].set_title(f"No-cycle ICL Contrasts - {checkpoint}\n{results['metadata']['model_name']}", fontsize=14)
        axes[col].set_xlabel("Attention Head", fontsize=12)
        axes[col].set_ylabel("Layer", fontsize=12)
        col += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved heatmap to {save_path}")
    
    plt.show()


def main():
    """Example usage of the multi-head analysis functions"""
    
    # Configuration matching your run_full_multihead_analysis.sh
    base_path = "/home/mmahaut/projects/parrots/outputs_multihead_full"
    model_name = "EleutherAI/pythia-1.4b"
    checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    n_cycles = 1
    max_length = 32
    cycle_range = [0, 1, 2, 3, 4, 5]  # Analyze cycles 0 through 5
    
    print("Loading multi-head analysis results...")
    results = load_multihead_results(
        base_path=base_path,
        model_name=model_name, 
        checkpoints=checkpoints,
        n_cycles=n_cycles,
        max_length=max_length
    )
    
    print(f"Loaded results for {len(results['natural'])} checkpoints")
    
    # Load results across cycles for each checkpoint
    print("Loading multi-head analysis results across cycles...")
    results_across_cycles = load_multihead_results_across_cycles(
        base_path=base_path,
        model_name=model_name,
        checkpoints=checkpoints,
        cycle_range=cycle_range,
        max_length=max_length
    )
    
    print(f"Loaded cycle evolution for {len(results_across_cycles['natural'])} checkpoints")
    
    # Create checkpoint evolution plot (across training steps)
    print("Creating checkpoint evolution plot...")
    plot_multihead_evolution(
        results, 
        save_path=f"{base_path}/multihead_checkpoint_evolution.png"
    )
    
    # Create cycle evolution plot (across cycles for each checkpoint)
    print("Creating cycle evolution plot...")
    plot_cycle_evolution_by_checkpoint(
        results_across_cycles,
        save_path=f"{base_path}/multihead_cycle_evolution.png"
    )
    
    # Create cycle summary statistics
    print("Creating cycle summary statistics...")
    plot_cycle_summary_statistics(
        results_across_cycles,
        save_path=f"{base_path}/multihead_cycle_summary.png"
    )
    
    # Create evolution plot
    print("Creating evolution plot...")
    plot_multihead_evolution(
        results, 
        save_path=f"{base_path}/multihead_evolution_analysis.png"
    )
    plot_multihead_evolution(
        results, 
        save_path=f"{base_path}/multihead_evolution_analysis.png"
    )
    
    # Create heatmap for latest checkpoint
    print("Creating checkpoint heatmap...")
    plot_checkpoint_heatmaps(
        results,
        checkpoint="steplatest",
        save_path=f"{base_path}/multihead_heatmap_steplatest.png"
    )
    
    # Create alluvial plot for repetition evolution
    print("Loading repetition evolution data...")
    repetition_data = load_repetition_evolution(
        base_path=base_path,
        model_name=model_name,
        checkpoints=checkpoints,
        n_cycles=n_cycles,
        max_length=max_length,
        target_layer=20  # Can be made configurable
    )
    
    if repetition_data:
        print("Creating alluvial plot...")
        create_alluvial_plot(
            repetition_data,
            save_path=f"{base_path}/repetition_alluvial_plot.png",
            target_layer=20
        )
    else:
        print("No repetition data found for alluvial plot")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for checkpoint in checkpoints:
        if checkpoint in results['natural']:
            natural_count = len(results['natural'][checkpoint])
            icl_count = len(results['icl'][checkpoint])
            print(f"  {checkpoint}: {natural_count} natural layers, {icl_count} ICL layers")


if __name__ == "__main__":
    main()


def load_mlp_results_across_cycles(base_path, model_name="EleutherAI/pythia-1.4b", checkpoints=None, 
                                  cycle_range=None, max_length=32):
    """
    Load MLP analysis results across both checkpoints and cycles from pipeline test outputs
    
    Args:
        base_path: Base output directory (e.g., "/home/mmahaut/projects/parrots/test_mlp_pipeline_output")
        model_name: Model name (for metadata only, directory structure is different)
        checkpoints: List of checkpoints to analyze
        cycle_range: List of cycle numbers to analyze (e.g., [0, 1, 2, 3, 4, 5])
        max_length: Maximum sequence length used in analysis
        
    Returns:
        dict: Contains results organized by checkpoint -> layer -> mlp_values
    """
    if checkpoints is None:
        checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    
    if cycle_range is None:
        cycle_range = list(range(6))  # Default: cycles 0-5
    
    base_dir = Path(base_path)
    
    results = {
        'natural': {},      # Natural data MLP contrasts
        'icl': {},         # ICL data MLP contrasts
        'no_cycle_icl': {}, # No-cycle ICL MLP contrasts
        'metadata': {
            'model_name': model_name,
            'cycle_range': cycle_range,
            'max_length': max_length,
            'checkpoints': checkpoints
        }
    }
    
    for checkpoint in checkpoints:
        checkpoint_dir = base_dir / checkpoint
        if not checkpoint_dir.exists():
            print(f"Warning: Checkpoint directory {checkpoint_dir} not found")
            continue
            
        results['natural'][checkpoint] = {}
        results['icl'][checkpoint] = {}
        results['no_cycle_icl'][checkpoint] = {}
        
        # Find all layer output files
        layer_files = list(checkpoint_dir.glob("pipeline_test_layer_*.out"))
        
        for layer_file in sorted(layer_files):
            # Extract layer number from filename
            layer_match = re.search(r'layer_(\d+)\.out', layer_file.name)
            if not layer_match:
                continue
            layer_idx = int(layer_match.group(1))
            
            try:
                # Parse the MLP results from the log file
                natural_value, icl_value, no_cycle_value = parse_mlp_log(layer_file, layer_idx)
                
                if natural_value is not None:
                    results['natural'][checkpoint][layer_idx] = natural_value
                if icl_value is not None:
                    results['icl'][checkpoint][layer_idx] = icl_value
                if no_cycle_value is not None:
                    results['no_cycle_icl'][checkpoint][layer_idx] = no_cycle_value
                    
            except Exception as e:
                print(f"Error parsing {layer_file}: {e}")
                continue
    
    return results


def parse_mlp_log(log_file, expected_layer):
    """
    Parse a single MLP log file to extract contrast values
    
    Args:
        log_file: Path to the log file
        expected_layer: Expected layer index for validation
        
    Returns:
        tuple: (natural_value, icl_value, no_cycle_value) as single float values or None
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract natural heatmap (single value)
    natural_pattern = rf"layer {expected_layer} natural heatmap: \[([^\]]+)\]"
    natural_match = re.search(natural_pattern, content)
    natural_value = None
    
    if natural_match:
        try:
            # Parse as single float value
            natural_str = natural_match.group(1).strip()
            natural_value = float(natural_str)
        except (ValueError, IndexError) as e:
            print(f"Error parsing natural heatmap for layer {expected_layer}: {e}")
    
    # Extract ICL heatmap (single value)
    icl_pattern = rf"layer {expected_layer} icl heatmap: \[([^\]]+)\]"
    icl_match = re.search(icl_pattern, content)
    icl_value = None
    
    if icl_match:
        try:
            # Parse as single float value
            icl_str = icl_match.group(1).strip()
            icl_value = float(icl_str)
        except (ValueError, IndexError) as e:
            print(f"Error parsing ICL heatmap for layer {expected_layer}: {e}")
    
    # Extract no-cycle ICL heatmap (may be None)
    no_cycle_pattern = rf"layer {expected_layer} no-cycle icl heatmap: (.+)"
    no_cycle_match = re.search(no_cycle_pattern, content)
    no_cycle_value = None
    
    if no_cycle_match:
        no_cycle_str = no_cycle_match.group(1).strip()
        if no_cycle_str != "None":
            try:
                # Remove brackets and parse as single float
                no_cycle_str = no_cycle_str.strip("[]")
                no_cycle_value = float(no_cycle_str)
            except (ValueError, IndexError) as e:
                print(f"Error parsing no-cycle ICL heatmap for layer {expected_layer}: {e}")
    
    return natural_value, icl_value, no_cycle_value


def create_mlp_dataframe(results_across_cycles, heatmap_type='natural'):
    """
    Convert MLP results into a pandas DataFrame
    
    Args:
        results_across_cycles: Results dictionary from load_mlp_results_across_cycles
        heatmap_type: 'natural', 'icl', or 'no_cycle_icl'
        
    Returns:
        pd.DataFrame: With columns ['checkpoint', 'layer', 'contrast']
    """
    data_rows = []
    
    for checkpoint, layers in results_across_cycles[heatmap_type].items():
        for layer_idx, contrast_value in layers.items():
            data_rows.append({
                'checkpoint': checkpoint,
                'layer': layer_idx,
                'contrast': contrast_value
            })
    
    return pd.DataFrame(data_rows)


def plot_mlp_evolution_by_checkpoint(results_across_cycles, save_path=None):
    """
    Create horizontal plots showing MLP contrast evolution across layers for each checkpoint,
    with ICL data in the top row and natural data in the bottom row
    
    Args:
        results_across_cycles: Results dictionary from load_mlp_results_across_cycles
        save_path: Path to save the plot (optional)
    """
    # Create DataFrames
    natural_df = create_mlp_dataframe(results_across_cycles, 'natural')
    icl_df = create_mlp_dataframe(results_across_cycles, 'icl')
    
    if icl_df.empty and natural_df.empty:
        print("No data available for plotting")
        return
    
    # Custom sorting function for checkpoints
    def sort_checkpoint_key(checkpoint):
        if checkpoint == 'steplatest':
            return float('inf')  # Always last
        elif checkpoint.startswith('step'):
            try:
                return int(checkpoint[4:])  # Extract number after 'step'
            except ValueError:
                return 0
        else:
            return 0
    
    # Filter checkpoints: skip step1 and step5000, keep others, sort by step number
    all_checkpoints = set()
    if not icl_df.empty:
        all_checkpoints.update(icl_df['checkpoint'].unique())
    if not natural_df.empty:
        all_checkpoints.update(natural_df['checkpoint'].unique())
    
    filtered_checkpoints = []
    for checkpoint in sorted(all_checkpoints, key=sort_checkpoint_key):
        # Skip step1 if it has insufficient data
        if checkpoint == 'step1':
            icl_data = icl_df[icl_df['checkpoint'] == checkpoint] if not icl_df.empty else pd.DataFrame()
            natural_data = natural_df[natural_df['checkpoint'] == checkpoint] if not natural_df.empty else pd.DataFrame()
            if len(icl_data) < 5 and len(natural_data) < 5:  # Skip if too little data
                print(f"Skipping {checkpoint} - insufficient data")
                continue
        # Skip step5000 and step7000 to save space in the figure
        if checkpoint in ['step5000', 'step7000']:
            print(f"Skipping {checkpoint} - excluded to save space")
            continue
        filtered_checkpoints.append(checkpoint)
    
    if len(filtered_checkpoints) == 0:
        print("No data available for plotting after filtering")
        return
    
    # Calculate shared y-axis limits for ALL plots (ICL + natural)
    all_contrast_values = []
    if not icl_df.empty:
        all_contrast_values.extend(icl_df['contrast'].values)
    if not natural_df.empty:
        all_contrast_values.extend(natural_df['contrast'].values)
    
    if len(all_contrast_values) > 0:
        y_min = min(all_contrast_values)
        y_max = max(all_contrast_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        y_min_padded = y_min - y_padding
        y_max_padded = y_max + y_padding
    else:
        y_min_padded, y_max_padded = 0, 1
    
    # Create figure with 2 rows (ICL top, Natural bottom)
    n_cols = len(filtered_checkpoints)
    fig_width = 4.5 * n_cols  # Adjust width based on number of checkpoints
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 8))  # 2 rows
    
    # Ensure axes is always 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    # Plot ICL contrasts in the top row
    for col, checkpoint in enumerate(filtered_checkpoints):
        checkpoint_icl = icl_df[icl_df['checkpoint'] == checkpoint] if not icl_df.empty else pd.DataFrame()
        
        if not checkpoint_icl.empty:
            # Sort by layer for proper line plot
            checkpoint_icl = checkpoint_icl.sort_values('layer')
            
            # Plot as line with single color
            axes[0, col].plot(checkpoint_icl['layer'], checkpoint_icl['contrast'], 
                            color='#1f77b4', linewidth=2, marker='o', markersize=4)
            
            axes[0, col].set_title(f"ICL MLP\n{checkpoint}", fontsize=14, fontweight='bold')
            axes[0, col].set_xlabel("Layer Number", fontsize=12)
            # Only show y-axis label on the first plot
            if col == 0:
                axes[0, col].set_ylabel("Contrast", fontsize=12)
            axes[0, col].tick_params(axis='both', which='major', labelsize=10)
            axes[0, col].set_ylim(y_min_padded, y_max_padded)  # Same scale for all
            axes[0, col].grid(True, alpha=0.3)
        else:
            # Empty plot if no data
            axes[0, col].text(0.5, 0.5, 'No ICL Data', ha='center', va='center', 
                            transform=axes[0, col].transAxes, fontsize=12)
            axes[0, col].set_title(f"ICL MLP\n{checkpoint}", fontsize=14, fontweight='bold')
            axes[0, col].set_xlabel("Layer Number", fontsize=12)
            if col == 0:
                axes[0, col].set_ylabel("Contrast", fontsize=12)
    
    # Plot Natural contrasts in the bottom row
    for col, checkpoint in enumerate(filtered_checkpoints):
        checkpoint_natural = natural_df[natural_df['checkpoint'] == checkpoint] if not natural_df.empty else pd.DataFrame()
        
        if not checkpoint_natural.empty:
            # Sort by layer for proper line plot
            checkpoint_natural = checkpoint_natural.sort_values('layer')
            
            # Plot as line with single color (different from ICL)
            axes[1, col].plot(checkpoint_natural['layer'], checkpoint_natural['contrast'], 
                            color='#ff7f0e', linewidth=2, marker='o', markersize=4)
            
            axes[1, col].set_title(f"Natural MLP\n{checkpoint}", fontsize=14, fontweight='bold')
            axes[1, col].set_xlabel("Layer Number", fontsize=12)
            # Only show y-axis label on the first plot
            if col == 0:
                axes[1, col].set_ylabel("Contrast", fontsize=12)
            axes[1, col].tick_params(axis='both', which='major', labelsize=10)
            axes[1, col].set_ylim(y_min_padded, y_max_padded)  # Same scale for all
            axes[1, col].grid(True, alpha=0.3)
        else:
            # Empty plot if no data
            axes[1, col].text(0.5, 0.5, 'No Natural Data', ha='center', va='center', 
                            transform=axes[1, col].transAxes, fontsize=12)
            axes[1, col].set_title(f"Natural MLP\n{checkpoint}", fontsize=14, fontweight='bold')
            axes[1, col].set_xlabel("Layer Number", fontsize=12)
            if col == 0:
                axes[1, col].set_ylabel("Contrast", fontsize=12)
    
    # Add a simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, marker='o', markersize=4, label='ICL'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, marker='o', markersize=4, label='Natural')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
              bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved horizontal MLP evolution plot to {save_path}")
    
    plt.show()


def main():
    """Example usage of the multi-head analysis functions"""
    
    # Configuration matching your run_full_multihead_analysis.sh
    base_path = "/home/mmahaut/projects/parrots/outputs_multihead_full"
    model_name = "EleutherAI/pythia-1.4b"
    checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
    
    print("Loading multi-head results across cycles...")
    results_across_cycles = load_multihead_results_across_cycles(
        base_path=base_path,
        model_name=model_name,
        checkpoints=checkpoints,
        cycle_range=list(range(6))  # Cycles 0-5
    )
    
    print("Creating cycle evolution plots...")
    plot_cycle_evolution_by_checkpoint(
        results_across_cycles, 
        save_path="/home/mmahaut/projects/parrots/cycle_evolution_horizontal.png"
    )
    
    # Optional: Create summary statistics plots
    print("Creating summary statistics plots...")
    plot_cycle_summary_statistics(
        results_across_cycles,
        save_path="/home/mmahaut/projects/parrots/cycle_summary_stats.png"
    )
    
    # Example of loading repetition evolution data
    print("Loading repetition evolution data...")
    repetition_data = load_repetition_evolution_from_outputs(
        base_path=base_path,
        model_name=model_name,
        checkpoints=checkpoints,
        layer=20,  # Focus on layer 20
        cycles=list(range(6))
    )
    
    # Create alluvial plot
    print("Creating alluvial plot...")
    create_alluvial_plot(
        repetition_data,
        save_path="/home/mmahaut/projects/parrots/repetition_alluvial.png",
        target_layer=20,
        paper_ready=True
    )
    
    print("Analysis complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate multihead analysis graphs')
    parser.add_argument('--base_path', type=str, required=False, help='Base path for analysis outputs')
    parser.add_argument('--model_name', type=str, required=False, help='Model name')
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory for graphs')
    
    args = parser.parse_args()
    
    # Use command-line arguments if provided, otherwise fall back to default main()
    if args.base_path and args.model_name and args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Configuration from command-line arguments
        base_path = args.base_path
        model_name = args.model_name
        output_dir = args.output_dir
        checkpoints = ["step1", "step1000", "step5000", "step7000", "step10000", "step100000", "steplatest"]
        n_cycles = 1
        max_length = 32
        cycle_range = [0, 1, 2, 3, 4, 5]
        
        print('Loading multi-head analysis results with 4-category support...')
        results = load_multihead_results(
            base_path=base_path,
            model_name=model_name, 
            checkpoints=checkpoints,
            n_cycles=n_cycles,
            max_length=max_length
        )
        print(f'Loaded results for {len(results["natural"])} checkpoints')
        print(f'Categories found: {list(results.keys())}')
        
        # Load results across cycles for each checkpoint
        print('Loading multi-head analysis results across cycles...')
        results_across_cycles = load_multihead_results_across_cycles(
            base_path=base_path,
            model_name=model_name,
            checkpoints=checkpoints,
            cycle_range=cycle_range,
            max_length=max_length
        )
        print(f'Loaded cycle evolution for {len(results_across_cycles["natural"])} checkpoints')
        
        # Create checkpoint evolution plot (across training steps)
        print('Creating checkpoint evolution plot with 4 categories...')
        plot_multihead_evolution(
            results, 
            save_path=f'{output_dir}/multihead_checkpoint_evolution_4cat.png'
        )
        
        # Create cycle evolution plot (across cycles for each checkpoint)
        print('Creating cycle evolution plot with 4 categories...')
        plot_cycle_evolution_by_checkpoint(
            results_across_cycles,
            save_path=f'{output_dir}/multihead_cycle_evolution_4cat.png'
        )
        
        # Create cycle summary statistics
        print('Creating cycle summary statistics with 4 categories...')
        plot_cycle_summary_statistics(
            results_across_cycles,
            save_path=f'{output_dir}/multihead_cycle_summary_4cat.png'
        )
        
        # Create heatmaps for all major checkpoints
        major_checkpoints = ['step1000', 'step10000', 'step100000', 'steplatest']
        for checkpoint in major_checkpoints:
            if checkpoint in results['natural']:
                print(f'Creating checkpoint heatmap for {checkpoint} with 4 categories...')
                plot_checkpoint_heatmaps(
                    results,
                    checkpoint=checkpoint,
                    save_path=f'{output_dir}/multihead_heatmap_{checkpoint}_4cat.png'
                )
        
        # Create alluvial plot for repetition evolution
        print('Loading repetition evolution data...')
        try:
            repetition_data = load_repetition_evolution(
                base_path=base_path,
                model_name=model_name,
                checkpoints=checkpoints,
                n_cycles=n_cycles,
                max_length=max_length,
                target_layer=20
            )
            
            if repetition_data:
                print('Creating alluvial plot...')
                create_alluvial_plot(
                    repetition_data,
                    save_path=f'{output_dir}/repetition_alluvial_plot_4cat.png',
                    target_layer=20
                )
            else:
                print('No repetition data found for alluvial plot')
        except Exception as e:
            print(f'Error creating alluvial plot: {e}')
        
        # Print summary statistics
        print('')
        print('Summary Statistics with 4-category support:')
        for checkpoint in checkpoints:
            if checkpoint in results['natural']:
                natural_count = len([x for x in results['natural'][checkpoint].values() if x is not None])
                icl_count = len([x for x in results['icl'][checkpoint].values() if x is not None])
                successful_icl_count = len([x for x in results['successful_icl'][checkpoint].values() if x is not None])
                no_cycle_icl_count = len([x for x in results['no_cycle_icl'][checkpoint].values() if x is not None])
                
                print(f'{checkpoint}:')
                print(f'  Natural: {natural_count} layers')
                print(f'  ICL: {icl_count} layers')
                print(f'  Successful ICL: {successful_icl_count} layers')
                print(f'  No-cycle ICL: {no_cycle_icl_count} layers')
        
        print('')
        print('Graph generation completed successfully!')
        print(f'All graphs saved to: {output_dir}')
        print('Generated files:')
        print('  - multihead_checkpoint_evolution_4cat.png')
        print('  - multihead_cycle_evolution_4cat.png')
        print('  - multihead_cycle_summary_4cat.png')
        print('  - multihead_heatmap_*_4cat.png (for major checkpoints)')
        print('  - repetition_alluvial_plot_4cat.png (if data available)')
    else:
        main()