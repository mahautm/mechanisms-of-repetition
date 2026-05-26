#!/usr/bin/env python3
"""
Beautiful Dual Alluvial Plot Generator
Creates subplot with no-cycle ICL data on left and natural data on right
Based on the original multihead_analysis_graphs.py logic with beautiful matplotlib styling
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import interpolate

class BeautifulDualAlluvial:
    """Beautiful dual alluvial generator with no-cycle ICL and natural data"""
    
    def get_interpretable_checkpoint_name(self, checkpoint, prefix=None):
        """Convert checkpoint name to interpretable short label
        
        Args:
            checkpoint: Checkpoint name like "step1000-tokens4B"
            prefix: Optional prefix like "PRE" for pre-training before instruction tuning
        """
        # Extract key info from checkpoint names
        # e.g., "step1000-tokens4B" -> "4B" or "PRE 4B"
        # e.g., "step343000-tokens1438B" -> "1438B" or "PRE 1438B"
        # e.g., "step1" -> "step1"
        
        if '-tokens' in checkpoint:
            # Extract token count
            token_part = checkpoint.split('-tokens')[1]
            # Add prefix if provided
            if prefix:
                return f"{prefix} {token_part}"
            # Remove 'B' and convert
            if token_part.endswith('B'):
                return token_part  # Keep as is like "4B", "1438B"
            elif token_part.endswith('T'):
                return token_part  # Keep as is like "15T"
            return token_part
        else:
            # Fallback for Pythia-style names
            base = checkpoint.replace('step', '')
            if prefix:
                return f"{prefix} {base}"
            return base
    
    def __init__(self):
        """Initialize with original paper colors and beautiful styling"""
        
        # Color palette for checkpoint progression
        self.checkpoint_colors = [
            '#4A90E2',   # Bright blue
            '#9B59B6',   # Vivid purple  
            '#E67E22',   # Bright orange
            '#E74C3C',   # Bright red
            '#27AE60',   # Bright green
            '#8B4513'    # Dark brown
        ]
        
        # Enhanced high-contrast colors for better visibility
        self.paper_colors = {
            "non_repeating": '#D0D0D0',           # Darker grey (more visible)
            "repeating_since_step1": '#4A90E2',   # Bright blue
            "repeating_since_step1000": '#9B59B6', # Vivid purple  
            "repeating_since_step5000": '#E67E22', # Bright orange
            "repeating_since_step10000": '#E74C3C', # Bright red
            "repeating_since_step100000": '#27AE60', # Bright green
            "repeating_since_steplatest": '#8B4513' # Dark brown
        }
        
        # Beautiful paper-ready styling with larger fonts
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'axes.linewidth': 0,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'xtick.bottom': False,
            'ytick.left': False,
            'axes.grid': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.facecolor': 'white'
        })
    
    def load_natural_repetition_data_from_logs(self, log_dir, job_checkpoint_map, layer, model_name="custom"):
        """Load natural repetition data from job log files (for OLMo/Apertus)"""
        
        repetition_data = {}
        log_path = Path(log_dir)
        
        for checkpoint, job_id in job_checkpoint_map.items():
            # Try different log file prefixes
            log_file = log_path / f"olmo_attention_{job_id}.out"
            
            if not log_file.exists():
                log_file = log_path / f"olmo_instruct_{job_id}.out"
            
            if not log_file.exists():
                log_file = log_path / f"apertus_attention_{job_id}.out"
            
            if not log_file.exists():
                print(f"Log file not found for checkpoint {checkpoint} (job {job_id})")
                continue
            
            try:
                with open(log_file, 'r') as f:
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
                        'cycle': 0,
                        'layer': layer
                    }
                    
                    print(f"Natural {checkpoint} layer {layer}: {len(data_indices)} datapoints, {len(repetition_indices)} repetitions")
                    
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")
                continue
        
        return repetition_data
    
    def load_natural_repetition_data(self, base_path, model_name="EleutherAI/pythia-1.4b", 
                                   checkpoints=None, layer=19, cycles=None, max_length=32):
        """Load natural (cycle 0) repetition evolution data"""
        
        model_path = Path(base_path) / model_name.replace("/", "/")
        
        if checkpoints is None:
            checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'step140000', 'step143000']
        
        repetition_data = {}
        
        for checkpoint in checkpoints:
            checkpoint_path = model_path / checkpoint / f"layer_{layer}"
            
            if not checkpoint_path.exists():
                print(f"Layer path not found: {checkpoint_path}")
                continue
            
            # Prefer the no-truncation output, but keep the legacy filename as a fallback.
            output_file = checkpoint_path / "full_analysis_cyc4_full.out"
            if not output_file.exists():
                output_file = checkpoint_path / f"full_analysis_cyc4_ml{max_length}.out"
            
            if output_file.exists():
                try:
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
                            'cycle': 0,
                            'layer': layer
                        }
                        
                        print(f"Natural {checkpoint} layer {layer}: {len(data_indices)} datapoints, {len(repetition_indices)} repetitions")
                        
                except Exception as e:
                    print(f"Error parsing {output_file}: {e}")
                    continue
        
        return repetition_data

    def load_no_cycle_icl_data_from_logs(self, log_dir, job_checkpoint_map, layer, model_name="custom"):
        """Load no-cycle ICL data from job log files (for OLMo/Apertus)
        
        Returns data in same format as Pythia:
        - data_indices: ALL unique datapoints that appear across ALL checkpoints
        - repetition_indices: which datapoints are repeating at THIS checkpoint
        """
        
        log_path = Path(log_dir)
        
        # First pass: collect all unique datapoint IDs across all checkpoints
        all_no_cycle_icl_datapoints = set()
        checkpoint_repetition_indices = {}
        
        for checkpoint, job_id in job_checkpoint_map.items():
            # Try different log file prefixes
            log_file = log_path / f"olmo_attention_{job_id}.out"
            
            if not log_file.exists():
                log_file = log_path / f"olmo_instruct_{job_id}.out"
            
            if not log_file.exists():
                log_file = log_path / f"apertus_attention_{job_id}.out"
            
            if not log_file.exists():
                print(f"Log file not found for {checkpoint}")
                checkpoint_repetition_indices[checkpoint] = []
                continue
            
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Extract no-cycle ICL indices
                no_cycle_match = re.search(rf'layer {layer} no-cycle icl index: \[(.*?)\]', content)
                
                if no_cycle_match:
                    no_cycle_str = no_cycle_match.group(1).strip()
                    
                    if no_cycle_str:
                        no_cycle_indices = [int(x.strip()) for x in no_cycle_str.split(',') if x.strip()]
                    else:
                        no_cycle_indices = []
                    
                    # Add these datapoint IDs to the global set
                    all_no_cycle_icl_datapoints.update(no_cycle_indices)
                    
                    # Store which datapoints are repeating at this checkpoint
                    checkpoint_repetition_indices[checkpoint] = no_cycle_indices
                    
                    print(f"No-cycle ICL {checkpoint} layer {layer}: {len(no_cycle_indices)} samples")
                else:
                    checkpoint_repetition_indices[checkpoint] = []
                    
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")
                checkpoint_repetition_indices[checkpoint] = []
                continue
        
        # Second pass: create data structure like Pythia
        # ALL datapoints appear in data_indices for each checkpoint
        # Only some are in repetition_indices (those that are repeating at that checkpoint)
        all_datapoints_sorted = sorted(all_no_cycle_icl_datapoints)
        
        repetition_data = {}
        for checkpoint in job_checkpoint_map.keys():
            if checkpoint in checkpoint_repetition_indices:
                repetition_data[checkpoint] = {
                    'data_indices': all_datapoints_sorted,  # ALL datapoints across all checkpoints
                    'repetition_indices': checkpoint_repetition_indices[checkpoint],  # Repeating at THIS checkpoint
                    'cycle': 0,
                    'layer': layer
                }
        
        return repetition_data
    
    def load_no_cycle_icl_data(self, base_path, model_name="EleutherAI/pythia-1.4b", 
                              checkpoints=None, layer=19, max_length=32):
        """Load no-cycle ICL repetition data from available cycles
        
        Returns data in same format as natural data:
        - data_indices: ALL unique datapoints that appear across ALL checkpoints
        - repetition_indices: which datapoints are repeating at THIS checkpoint
        """
        
        model_path = Path(base_path) / model_name.replace("/", "/")
        
        if checkpoints is None:
            checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'step140000', 'step143000']
        
        # First pass: collect all unique datapoint IDs across all checkpoints
        all_no_cycle_icl_datapoints = set()
        checkpoint_repetition_indices = {}
        
        for checkpoint in checkpoints:
            checkpoint_path = model_path / checkpoint / f"layer_{layer}"
            
            if not checkpoint_path.exists():
                print(f"Layer path not found: {checkpoint_path}")
                continue
            
            # Try cycles 0-4 to find no-cycle ICL data
            found_data = False
            for cycle in range(5):
                output_file = checkpoint_path / f"full_analysis_cyc{cycle}_full.out"
                if not output_file.exists():
                    output_file = checkpoint_path / f"full_analysis_cyc{cycle}_ml{max_length}.out"
                
                if output_file.exists():
                    try:
                        with open(output_file, 'r') as f:
                            content = f.read()
                        
                        # Extract no-cycle ICL indices and cycle count
                        # Try both naming patterns: "no-cycle icl index" and "no-cycle index"
                        no_cycle_icl_match = re.search(rf'layer {layer} no-cycle icl index: \[(.*?)\]', content)
                        if not no_cycle_icl_match:
                            # Try alternative naming (used in steplatest)
                            no_cycle_icl_match = re.search(rf'layer {layer} no-cycle index: \[(.*?)\]', content)
                        
                        no_cycle_icl_count_match = re.search(rf'layer {layer} no-cycle icl cycle count: (\d+)', content)
                        
                        if no_cycle_icl_match and no_cycle_icl_count_match:
                            no_cycle_icl_str = no_cycle_icl_match.group(1).strip()
                            no_cycle_icl_count = int(no_cycle_icl_count_match.group(1))
                            
                            if no_cycle_icl_count == 0:
                                # Empty checkpoint - no repetitions
                                checkpoint_repetition_indices[checkpoint] = []
                                print(f"No-cycle ICL {checkpoint} layer {layer} cycle {cycle}: {no_cycle_icl_count} samples (empty)")
                            else:
                                # Parse indices for non-zero counts
                                no_cycle_icl_indices = [int(x.strip()) for x in no_cycle_icl_str.split(',') if x.strip()]
                                
                                # Add these datapoint IDs to the global set
                                all_no_cycle_icl_datapoints.update(no_cycle_icl_indices)
                                
                                # Store which datapoints are repeating at this checkpoint
                                checkpoint_repetition_indices[checkpoint] = no_cycle_icl_indices
                                
                                print(f"No-cycle ICL {checkpoint} layer {layer} cycle {cycle}: {no_cycle_icl_count} samples, {len(no_cycle_icl_indices)} repetitions")
                            
                            found_data = True
                            break
                            
                    except Exception as e:
                        print(f"Error parsing {output_file}: {e}")
                        continue
            
            if not found_data:
                print(f"No no-cycle ICL data found for {checkpoint} layer {layer}")
                # Still add empty entry for consistency
                checkpoint_repetition_indices[checkpoint] = []
        
        # Second pass: create data structure like natural data
        # ALL datapoints appear in data_indices for each checkpoint
        # Only some are in repetition_indices (those that are repeating at that checkpoint)
        all_datapoints_sorted = sorted(all_no_cycle_icl_datapoints)
        
        no_cycle_icl_data = {}
        for checkpoint in checkpoints:
            if checkpoint in checkpoint_repetition_indices:
                no_cycle_icl_data[checkpoint] = {
                    'data_indices': all_datapoints_sorted,  # ALL datapoints across all checkpoints
                    'repetition_indices': checkpoint_repetition_indices[checkpoint],  # Repeating at THIS checkpoint
                    'cycle': 0,
                    'layer': layer
                }
        
        
        return no_cycle_icl_data

    def create_progressive_categorization(self, repetition_data, checkpoints):
        """Create progressive categorization exactly like the original"""
        
        # Get all datapoints across all checkpoints
        all_datapoints = set()
        for checkpoint_data in repetition_data.values():
            all_datapoints.update(checkpoint_data['data_indices'])
        all_datapoints = sorted(all_datapoints)
        
        print(f"Processing {len(all_datapoints)} total datapoints")
        
        # Determine repetitive/non-repetitive status for each datapoint at each checkpoint
        datapoint_states = {}  # checkpoint -> datapoint -> is_repeating
        
        for checkpoint in checkpoints:
            if checkpoint not in repetition_data:
                continue
            
            datapoint_states[checkpoint] = {}
            repetition_indices_set = set(repetition_data[checkpoint]['repetition_indices'])
            
            for dp in all_datapoints:
                if dp in repetition_data[checkpoint]['data_indices']:
                    datapoint_states[checkpoint][dp] = dp in repetition_indices_set
        
        # Progressive categorization - track when each datapoint first became repetitive
        datapoint_progressive_categories = {}
        
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
                    # Find when this datapoint first became repetitive
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
        
        return datapoint_progressive_categories, all_datapoints

    def draw_beautiful_flow(self, ax, x1, x2, y1_start, y1_end, y2_start, y2_end, color, alpha=0.6):
        """Draw a beautiful smooth alluvial flow using Bezier curves"""
        
        # Calculate control points for smooth Bezier curve
        mid_x = (x1 + x2) / 2
        
        # Create smooth path using cubic Bezier curves
        path_data = [
            # Top curve
            (mpath.Path.MOVETO, (x1, y1_end)),
            (mpath.Path.CURVE4, (mid_x, y1_end)),
            (mpath.Path.CURVE4, (mid_x, y2_end)),
            (mpath.Path.CURVE4, (x2, y2_end)),
            # Right edge
            (mpath.Path.LINETO, (x2, y2_start)),
            # Bottom curve (reverse)
            (mpath.Path.CURVE4, (mid_x, y2_start)),
            (mpath.Path.CURVE4, (mid_x, y1_start)),
            (mpath.Path.CURVE4, (x1, y1_start)),
            # Close path
            (mpath.Path.CLOSEPOLY, (x1, y1_end))
        ]
        
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        
        # Create beautiful patch with gradient-like appearance
        patch = PathPatch(path, facecolor=color, alpha=alpha, 
                         edgecolor='none', linewidth=0)
        ax.add_patch(patch)
        
        # Add subtle edge highlight for depth
        edge_patch = PathPatch(path, facecolor='none', 
                              edgecolor=color, alpha=alpha*0.8, linewidth=0.3)
        ax.add_patch(edge_patch)

    def create_subplot_alluvial(self, ax, repetition_data, title_suffix, checkpoints, show_ylabel=True, label_prefix=None):
        """Create alluvial plot in given axis
        
        Args:
            show_ylabel: If False, don't show y-axis label (for right subplot)
            label_prefix: Optional prefix for X-axis labels (e.g., "PRE" for pre-training)
        """
        
        # Use ALL checkpoints, even if some don't have data
        # We'll handle empty checkpoints gracefully in the plotting logic
        available_checkpoints = checkpoints
        
        # Check if we have at least some data
        checkpoints_with_data = [cp for cp in checkpoints if cp in repetition_data and len(repetition_data[cp]['data_indices']) > 0]
        if len(checkpoints_with_data) < 2:
            print(f"Need at least 2 checkpoints with data for alluvial plot ({title_suffix})")
            return None
        
        # Get progressive categorization using original logic
        datapoint_progressive_categories, all_datapoints = self.create_progressive_categorization(
            repetition_data, available_checkpoints
        )
        
        # Count datapoints by progressive category at each checkpoint
        progressive_category_counts = {}
        all_categories = set()
        
        for checkpoint in available_checkpoints:
            # Skip checkpoints without data in the categorization data
            if checkpoint not in datapoint_progressive_categories:
                # But still initialize empty counts to maintain x-axis positions
                progressive_category_counts[checkpoint] = defaultdict(int)
                continue
            
            progressive_category_counts[checkpoint] = defaultdict(int)
            
            for dp in all_datapoints:
                if dp in datapoint_progressive_categories[checkpoint]:
                    category = datapoint_progressive_categories[checkpoint][dp]
                    progressive_category_counts[checkpoint][category] += 1
                    all_categories.add(category)
        
        checkpoint_positions = np.arange(len(available_checkpoints))
        bar_width = 0.6
        spacing = 1.0
        
        # Calculate total counts for proportion calculation
        total_counts = {}
        for checkpoint in available_checkpoints:
            if checkpoint in progressive_category_counts and sum(progressive_category_counts[checkpoint].values()) > 0:
                total_counts[checkpoint] = sum(progressive_category_counts[checkpoint].values())
            else:
                # No data for this checkpoint - will show as empty bar
                total_counts[checkpoint] = 0
        
        # Track bar positions for flow drawing
        bar_positions = {}
        
        # Draw stacked bars for each checkpoint
        for i, checkpoint in enumerate(available_checkpoints):
            x_pos = i * spacing
            current_y = 0
            
            # Handle empty checkpoints - skip drawing but keep position
            if checkpoint not in progressive_category_counts or total_counts[checkpoint] == 0:
                # Draw empty placeholder bar (very thin grey bar to show position)
                rect = patches.Rectangle(
                    (x_pos - bar_width/2, 0), bar_width, 0.02,
                    facecolor='#E0E0E0', alpha=0.3,
                    edgecolor='grey', linewidth=0.5, linestyle='--'
                )
                ax.add_patch(rect)
                continue
            
            # Sort categories to ensure consistent ordering
            sorted_categories = sorted([cat for cat in all_categories 
                                     if progressive_category_counts[checkpoint][cat] > 0])
            
            for category in sorted_categories:
                count = progressive_category_counts[checkpoint][category]
                
                if count > 0:
                    # Convert to proportion for y-axis
                    total = total_counts[checkpoint]
                    proportion = count / total
                    
                    # Draw beautiful bar segment with gradient effect
                    color = self.paper_colors.get(category, '#D0D0D0')
                    
                    # Main bar with higher opacity for better visibility
                    rect = patches.Rectangle(
                        (x_pos - bar_width/2, current_y), bar_width, proportion,
                        facecolor=color, alpha=0.9,
                        edgecolor='black', linewidth=0.8
                    )
                    ax.add_patch(rect)
                    
                    # Subtle gradient effect
                    gradient_rect = patches.Rectangle(
                        (x_pos - bar_width/2, current_y), bar_width/3, proportion,
                        facecolor='white', alpha=0.2,
                        edgecolor='none'
                    )
                    ax.add_patch(gradient_rect)
                    
                    # Add proportion text inside non-repeating bars
                    if category == "non_repeating" and proportion > 0.05:  # Only if bar is large enough
                        ax.text(x_pos, current_y + proportion/2, f'{proportion:.2f}',
                               ha='center', va='center', fontweight='bold', 
                               fontsize=10, color='#333333')
                    
                    # Store position for flow drawing
                    bar_positions[(checkpoint, category)] = (current_y, current_y + proportion)
                    
                    current_y += proportion
        
        # Draw beautiful flows between consecutive checkpoints
        for i in range(len(available_checkpoints) - 1):
            cp1 = available_checkpoints[i]
            cp2 = available_checkpoints[i + 1]
            
            x1 = i * spacing + bar_width/2
            x2 = (i + 1) * spacing - bar_width/2
            
            # Track datapoint transitions
            transitions = defaultdict(int)
            
            if (cp1 in datapoint_progressive_categories and 
                cp2 in datapoint_progressive_categories):
                
                for dp in all_datapoints:
                    if (dp in datapoint_progressive_categories[cp1] and 
                        dp in datapoint_progressive_categories[cp2]):
                        
                        cat1 = datapoint_progressive_categories[cp1][dp]
                        cat2 = datapoint_progressive_categories[cp2][dp]
                        transitions[(cat1, cat2)] += 1
            
            # Draw flows for each transition
            for (cat1, cat2), count in transitions.items():
                if count > 3 and (cp1, cat1) in bar_positions and (cp2, cat2) in bar_positions:  # Lower threshold for no-cycle ICL
                    
                    y1_start, y1_end = bar_positions[(cp1, cat1)]
                    y2_start, y2_end = bar_positions[(cp2, cat2)]
                    
                    # Calculate proportional positioning within the bars
                    flow_height1 = (y1_end - y1_start) * (count / progressive_category_counts[cp1][cat1])
                    flow_height2 = (y2_end - y2_start) * (count / progressive_category_counts[cp2][cat2])
                    
                    # Use middle portion of the bars for flow
                    y1_mid = (y1_start + y1_end) / 2
                    y2_mid = (y2_start + y2_end) / 2
                    
                    flow_y1_start = y1_mid - flow_height1/2
                    flow_y1_end = y1_mid + flow_height1/2
                    flow_y2_start = y2_mid - flow_height2/2
                    flow_y2_end = y2_mid + flow_height2/2
                    
                    # Get flow color (use source category color)
                    flow_color = self.paper_colors.get(cat1, '#D0D0D0')
                    
                    # Draw beautiful flow
                    self.draw_beautiful_flow(
                        ax, x1, x2, 
                        flow_y1_start, flow_y1_end,
                        flow_y2_start, flow_y2_end,
                        flow_color, alpha=0.4
                    )
        
        # Beautiful styling
        ax.set_xlim(-0.3, len(available_checkpoints) - 0.7)
        ax.set_ylim(-0.02, 1.02)
        
        # Clean checkpoint labels using interpretable names with optional prefix
        checkpoint_labels = [self.get_interpretable_checkpoint_name(cp, prefix=label_prefix) for cp in available_checkpoints]
        ax.set_xticks(checkpoint_positions * spacing)
        ax.set_xticklabels(checkpoint_labels, fontsize=10, fontweight='bold', rotation=45, ha='right')
        
        # Y-axis styling for proportions - only show label if requested
        if show_ylabel:
            ax.set_ylabel('Proportion of Datapoints', fontsize=12, fontweight='bold', color='#333333')
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=10)
        
        # Subplot title
        ax.text(0.5, 1.05, title_suffix, transform=ax.transAxes, 
               ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # Return categories for legend
        return all_categories

    def create_beautiful_dual_alluvial(self, natural_data, no_cycle_icl_data, output_path, checkpoints=None, label_prefix=None):
        """Create beautiful dual alluvial plot with both datasets
        
        Args:
            natural_data: Natural repetition data
            no_cycle_icl_data: No-cycle ICL data
            output_path: Path to save plot
            checkpoints: List of checkpoint names
            label_prefix: Optional prefix for X-axis labels (e.g., "PRE" for pre-training before instruction tuning)
        """
        print("Creating beautiful dual alluvial plot...")
        
        if checkpoints is None:
            checkpoints = ['step1', 'step1000', 'step5000', 'step10000', 'step100000', 'step140000', 'step143000']
        
        print(f"Using checkpoints: {checkpoints}")
        print(f"Natural data keys: {list(natural_data.keys()) if natural_data else 'None'}")
        print(f"No-cycle ICL data keys: {list(no_cycle_icl_data.keys()) if no_cycle_icl_data else 'None'}")
        
        # Dynamically assign colors to checkpoints and create readable labels
        self.checkpoint_labels = {}
        for i, checkpoint in enumerate(checkpoints):
            category_key = f"repeating_since_{checkpoint}"
            if category_key not in self.paper_colors:
                color_idx = i % len(self.checkpoint_colors)
                self.paper_colors[category_key] = self.checkpoint_colors[color_idx]
                print(f"Assigned color {self.checkpoint_colors[color_idx]} to {category_key}")
            
            # Create readable label
            if 'tokens' in checkpoint:
                # OLMo/Apertus format: step1000-tokens4B -> "4B tokens"
                tokens_part = checkpoint.split('tokens')[1]
                self.checkpoint_labels[category_key] = f"Since {tokens_part}"
            elif checkpoint == 'steplatest':
                self.checkpoint_labels[category_key] = "Since final"
            elif checkpoint.startswith('step'):
                # Pythia format: step1000 -> "Step 1K"
                step_num = checkpoint.replace('step', '')
                if len(step_num) >= 4:
                    step_num = step_num[:-3] + 'K'
                self.checkpoint_labels[category_key] = f"Since step {step_num}"
            else:
                self.checkpoint_labels[category_key] = f"Since {checkpoint}"
        
        # Create dual subplot layout - larger to avoid overlap
        fig_width = 14
        fig_height = 6
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        
        # Create no-cycle ICL plot on left
        print("\nProcessing No-Cycle ICL data...")
        no_cycle_categories = self.create_subplot_alluvial(
            ax1, no_cycle_icl_data, "ICL", checkpoints, show_ylabel=True, label_prefix=label_prefix
        )
        
        # Create natural plot on right (no y-axis label)
        print("\nProcessing Natural data...")
        natural_categories = self.create_subplot_alluvial(
            ax2, natural_data, "Natural", checkpoints, show_ylabel=False, label_prefix=label_prefix
        )
        
        # Combine all categories for legend
        all_categories = set()
        if no_cycle_categories:
            all_categories.update(no_cycle_categories)
        if natural_categories:
            all_categories.update(natural_categories)
        
        # Beautiful shared legend
        if all_categories:
            legend_elements = []
            # Merge hardcoded labels with dynamic labels
            category_labels = {
                "non_repeating": "Non-repeating",
                "repeating_since_step1": "Repeating since step 1", 
                "repeating_since_step1000": "Repeating since step 1K",
                "repeating_since_step5000": "Repeating since step 5K",
                "repeating_since_step10000": "Repeating since step 10K", 
                "repeating_since_step100000": "Repeating since step 100K",
                "repeating_since_steplatest": "Repeating since latest"
            }
            # Add dynamically generated labels
            category_labels.update(self.checkpoint_labels)
            
            for category in sorted(all_categories):
                legend_elements.append(
                    patches.Patch(facecolor=self.paper_colors.get(category, '#D0D0D0'), 
                                alpha=0.8, edgecolor='white', linewidth=1,
                                label=category_labels.get(category, category))
                )
            
            # Place legend at bottom - compact for paper, closer to figures
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.01), 
                      ncol=4, fontsize=9, frameon=True, 
                      fancybox=False, shadow=False, columnspacing=1.0, handletextpad=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.16)  # Reduced from 0.18 to bring legend closer
        
        # Save in multiple formats
        base_path = Path(output_path).with_suffix('')
        for fmt in ['png', 'pdf']:
            save_path = f"{base_path}_dual_alluvial.{fmt}"
            plt.savefig(save_path, format=fmt, bbox_inches='tight', pad_inches=0.1,
                       facecolor='white', edgecolor='none')
            print(f"Dual plot saved: {save_path}")
        
        plt.close()
        return f"{base_path}_dual_alluvial.png"

def main():
    """Generate beautiful dual alluvial plots"""
    print("Beautiful Dual Alluvial Generator")
    print("=" * 33)
    
    # Configuration - can be modified or made into arguments
    log_base = Path("/home/mmahaut/projects/parrots/outputs_multihead_full_new")
    
    # Process both models
    model_configs = [
        {"model": "EleutherAI/pythia-1.4b", "layer": 23},  # 24 layers, use layer 19
    ]
    
    generator = BeautifulDualAlluvial()
    
    for config in model_configs:
        model = config["model"]
        layer = config["layer"]
        
        print(f"\nProcessing {model} (layer {layer})...")
        
        # Create model-specific output filename
        model_short = model.split("/")[-1]
        output_file = log_base / f"alluvial_{model_short}_layer_{layer}_dual.png"
        
        # Load natural data (cycle 0)
        print("Loading natural repetition data...")
        natural_data = generator.load_natural_repetition_data(
            log_base, model, layer=layer
        )
        
        # Load no-cycle ICL data
        print("Loading no-cycle ICL data...")
        no_cycle_icl_data = generator.load_no_cycle_icl_data(
            log_base, model, layer=layer
        )
        
        if not natural_data and not no_cycle_icl_data:
            print(f"No data found for {model}")
            continue
        
        result_path = generator.create_beautiful_dual_alluvial(
            natural_data, no_cycle_icl_data, output_file
        )
        if result_path:
            print(f"✓ Generated dual plot: {result_path}")

if __name__ == "__main__":
    main()